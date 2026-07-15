"""Dependency-injection seam.

A :class:`Services` container bundles the collaborators the API needs. Building
them in one place (``build_services``) lets the app wire real implementations at
startup while tests substitute fakes via FastAPI ``dependency_overrides``.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.cache import SemanticCache
from app.config import Settings
from app.embeddings import Embedder
from app.llm import LLMClient
from app.metrics import Metrics
from app.router import ComplexityRouter


@dataclass
class Services:
    """All runtime collaborators used by the request handlers."""

    settings: Settings
    embedder: Embedder
    cache: SemanticCache
    router: ComplexityRouter
    llm: LLMClient
    metrics: Metrics


def build_services(settings: Settings) -> Services:
    """Construct real Services from settings (called during app startup)."""
    embedder = Embedder(settings.embedding_model)
    cache = SemanticCache(
        collection_name=settings.chroma_collection,
        similarity_threshold=settings.similarity_threshold,
        use_remote=settings.use_remote_chroma,
        host=settings.chroma_host or "localhost",
        port=settings.chroma_port,
        persist_dir=settings.chroma_persist_dir,
    )
    router = ComplexityRouter(settings.complexity_word_threshold)
    llm = LLMClient(
        ollama_host=settings.ollama_host,
        ollama_model=settings.ollama_model,
        ollama_timeout_s=settings.ollama_timeout_s,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
    )
    metrics = Metrics(
        remote_input_cost_per_1m=settings.remote_input_cost_per_1m,
        remote_output_cost_per_1m=settings.remote_output_cost_per_1m,
        local_input_cost_per_1m=settings.local_input_cost_per_1m,
        local_output_cost_per_1m=settings.local_output_cost_per_1m,
        monthly_query_volume=settings.projected_monthly_queries,
    )
    return Services(
        settings=settings,
        embedder=embedder,
        cache=cache,
        router=router,
        llm=llm,
        metrics=metrics,
    )


def get_services(request: Request) -> Services:
    """FastAPI dependency returning the app-scoped Services container.

    Overridden in tests via ``app.dependency_overrides[get_services]``.
    """
    return request.app.state.services
