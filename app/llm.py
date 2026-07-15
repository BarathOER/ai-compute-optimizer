"""LLM backends: a local Ollama model and the remote Gemini API.

Both backends are exposed through a single :class:`LLMClient` so callers only
deal with ``generate(prompt, route)``. Heavy SDK imports are deferred to call
time to keep module import cheap and test-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.router import Route


@dataclass(frozen=True)
class LLMResult:
    """A generated completion plus the model that produced it."""

    text: str
    model: str


class LLMClient:
    """Dispatches generation to Ollama (local) or Gemini (remote)."""

    def __init__(
        self,
        *,
        ollama_host: str,
        ollama_model: str,
        ollama_timeout_s: float,
        gemini_api_key: str | None,
        gemini_model: str,
    ) -> None:
        self._ollama_host = ollama_host.rstrip("/")
        self._ollama_model = ollama_model
        self._ollama_timeout_s = ollama_timeout_s
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model

    async def generate(self, prompt: str, route: Route) -> LLMResult:
        """Generate a completion using the backend chosen by ``route``."""
        if route == "local":
            return await self._call_ollama(prompt)
        return await self._call_gemini(prompt)

    async def _call_ollama(self, prompt: str) -> LLMResult:
        """Call a local Ollama server via its HTTP generate endpoint."""
        url = f"{self._ollama_host}/api/generate"
        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._ollama_timeout_s) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return LLMResult(text=data.get("response", ""), model=self._ollama_model)

    async def _call_gemini(self, prompt: str) -> LLMResult:
        """Call the Gemini API via the google-generativeai SDK."""
        if not self._gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; cannot route to the remote model."
            )

        # Deferred import keeps the SDK optional for local/local-only setups.
        import google.generativeai as genai

        genai.configure(api_key=self._gemini_api_key)
        model = genai.GenerativeModel(self._gemini_model)
        # The SDK call is synchronous; run it off the event loop.
        import anyio

        response = await anyio.to_thread.run_sync(model.generate_content, prompt)
        return LLMResult(text=response.text, model=self._gemini_model)
