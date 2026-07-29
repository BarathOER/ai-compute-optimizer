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

    # --- Two-stage retrieval --------------------------------------------
    # Stage 1: bi-encoder cosine recall filter. Tuned for RECALL, not
    # precision — it only decides which cached prompts are worth reranking.
    stage1_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Min cosine similarity for a cached prompt to become a "
        "stage-2 rerank candidate.",
    )
    rerank_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of nearest cached prompts retrieved for reranking.",
    )
    # Stage 2: cross-encoder reranker. See app/reranker.py — the threshold is an
    # empirical cut point for this model, not a calibrated probability.
    enable_reranker: bool = Field(
        default=True,
        description="If false, fall back to bi-encoder-only (stage 1) matching. "
        "Lets both architectures be A/B compared.",
    )
    reranker_model: str = Field(default="cross-encoder/quora-distilroberta-base")
    reranker_threshold: float = Field(
        default=0.943,
        description="Min cross-encoder score to accept a cache hit (empirical, "
        "model-specific — re-tune via eval/two_stage_eval.py).",
    )

    # --- Router ----------------------------------------------------------
    complexity_word_threshold: int = Field(
        default=40,
        description="Prompts longer than this many words are treated as "
        "complex and routed to the remote model.",
    )
    enable_local_route: bool = Field(
        default=True,
        description="If false, every cache miss is routed to the remote model "
        "(Gemini) instead of the local one (Ollama). Set false for cloud "
        "deploys with no Ollama; the two-stage cache is unaffected.",
    )

    # --- Ollama (local model) -------------------------------------------
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_timeout_s: float = Field(default=60.0)

    # --- Gemini (remote model) ------------------------------------------
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")

    # --- Cost model (USD per 1M tokens) ---------------------------------
    # Real LLM pricing bills input (prompt) and output (completion) tokens at
    # different rates; output is materially more expensive. Rates are per 1M
    # tokens to match how providers publish list prices. Defaults reflect
    # Gemini 3.5 Flash list pricing (verified July 2026). Local inference is
    # treated as free at the token level.
    remote_input_cost_per_1m: float = Field(default=1.50)
    remote_output_cost_per_1m: float = Field(default=9.00)
    local_input_cost_per_1m: float = Field(default=0.0)
    local_output_cost_per_1m: float = Field(default=0.0)

    # --- Savings projection ---------------------------------------------
    # Expected production query volume, used to project monthly/annual savings
    # from the measured hit rate and average per-query token usage.
    projected_monthly_queries: int = Field(
        default=100_000,
        ge=0,
        description="Monthly query volume assumed when projecting savings.",
    )

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
