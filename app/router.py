"""Complexity-based routing.

Decides whether a prompt is "simple" (cheap local model) or "complex" (remote
frontier model). The heuristic is intentionally transparent and cheap: it
combines prompt length with the presence of reasoning-heavy keywords. Swap in a
learned classifier later without changing callers.
"""

from __future__ import annotations

from typing import Literal

Route = Literal["local", "remote"]

# Keywords that strongly suggest multi-step reasoning, generation, or analysis.
_COMPLEX_KEYWORDS: frozenset[str] = frozenset(
    {
        "analyze",
        "analyse",
        "compare",
        "contrast",
        "design",
        "architect",
        "explain why",
        "prove",
        "derive",
        "optimize",
        "refactor",
        "debug",
        "summarize",
        "write code",
        "implement",
        "step by step",
        "reasoning",
        "strategy",
    }
)


class ComplexityRouter:
    """Routes prompts to a local or remote model by estimated complexity."""

    def __init__(self, word_threshold: int, enable_local_route: bool = True) -> None:
        self._word_threshold = word_threshold
        self._enable_local_route = enable_local_route

    def route(self, prompt: str) -> Route:
        """Return the backend for a cache-miss query.

        Normally ``"local"`` for simple prompts and ``"remote"`` for complex
        ones. When the local route is disabled (e.g. a cloud deploy with no
        Ollama), *every* miss goes ``"remote"`` so the gateway still works.
        """
        if not self._enable_local_route:
            return "remote"
        return "remote" if self.is_complex(prompt) else "local"

    def is_complex(self, prompt: str) -> bool:
        """Heuristically decide whether a prompt is complex."""
        normalized = prompt.lower()
        word_count = len(prompt.split())

        if word_count > self._word_threshold:
            return True
        if any(keyword in normalized for keyword in _COMPLEX_KEYWORDS):
            return True
        # Multiple questions in one prompt tend to need stronger models.
        if normalized.count("?") >= 2:
            return True
        return False
