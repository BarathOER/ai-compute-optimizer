"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming prompt to be answered (possibly from cache)."""

    prompt: str = Field(..., min_length=1, description="The user prompt.")
    force_refresh: bool = Field(
        default=False,
        description="If true, bypass the semantic cache and always call an LLM.",
    )


class QueryResponse(BaseModel):
    """Answer plus the metadata a cost-conscious caller cares about."""

    answer: str
    cache_hit: bool = Field(description="True if served from the semantic cache.")
    route: Literal["cache", "local", "remote"] = Field(
        description="Where the answer came from."
    )
    model: str = Field(description="Model that produced the answer.")
    similarity: float | None = Field(
        default=None,
        description="Stage-1 cosine similarity of the nearest cache entry "
        "(reported on hits and misses for diagnosability).",
    )
    reranker_score: float | None = Field(
        default=None,
        description="Stage-2 cross-encoder score of the best candidate, if the "
        "reranker ran (not a calibrated probability).",
    )
    latency_ms: float = Field(description="End-to-end server latency.")
    estimated_cost_usd: float = Field(
        description="Estimated token cost of this request (0.0 on a cache hit)."
    )
    estimated_savings_usd: float = Field(
        description="Estimated cost avoided versus always calling the remote model."
    )


class HealthResponse(BaseModel):
    """Liveness/readiness payload for ``/health``."""

    status: Literal["ok"] = "ok"
    app: str
    version: str
    cache_size: int = Field(description="Number of entries in the semantic cache.")
