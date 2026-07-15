"""Application configuration.

All settings are loaded from environment variables (or an optional local
``.env`` file). Secrets such as API keys are never hardcoded — they are read
from the environment at runtime. See ``.env.example`` for the full list.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Service ---------------------------------------------------------
    app_name: str = Field(default="AI Compute Optimizer")
    log_level: str = Field(default="INFO")

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model used to embed prompts.",
    )

    # --- Semantic cache (ChromaDB) --------------------------------------
    chroma_host: str | None = Field(
        default=None,
        description="Host of a remote Chroma server. If unset, a local "
        "persistent client at ``chroma_persist_dir`` is used.",
    )
    chroma_port: int = Field(default=8000)
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection: str = Field(default="semantic_cache")
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for a prompt to count as a "
        "cache hit.",
    )

    # --- Router ----------------------------------------------------------
    complexity_word_threshold: int = Field(
        default=40,
        description="Prompts longer than this many words are treated as "
        "complex and routed to the remote model.",
    )

    # --- Ollama (local model) -------------------------------------------
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_timeout_s: float = Field(default=60.0)

    # --- Gemini (remote model) ------------------------------------------
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")

    # --- Cost model (USD per 1K tokens) ---------------------------------
    # Local inference is treated as free at the token level; the remote
    # model carries a per-1K-token price used for savings estimates.
    local_cost_per_1k_tokens: float = Field(default=0.0)
    remote_cost_per_1k_tokens: float = Field(default=0.0005)

    @property
    def use_remote_chroma(self) -> bool:
        """Whether a remote Chroma HTTP server is configured."""
        return self.chroma_host is not None


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so the environment is parsed once per process. Tests may clear the
    cache via ``get_settings.cache_clear()``.
    """
    return Settings()
