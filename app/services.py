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
from app.reranker import Reranker
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
    reranker: Reranker | None = None


def build_services(settings: Settings) -> Services:
    """Construct real Services from settings (called during app startup).

    The cross-encoder reranker is loaded here, once, so its (multi-hundred-MB)
    weights are initialized at startup rather than on the first request.
    """
    embedder = Embedder(settings.embedding_model)
    reranker = (
        Reranker(settings.reranker_model, settings.reranker_threshold)
        if settings.enable_reranker
        else None
    )
    cache = SemanticCache(
        collection_name=settings.chroma_collection,
        stage1_threshold=settings.stage1_threshold,
        top_k=settings.rerank_top_k,
        enable_reranker=settings.enable_reranker,
        reranker=reranker,
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
        reranker=reranker,
    )


def get_services(request: Request) -> Services:
    """FastAPI dependency returning the app-scoped Services container.

    Overridden in tests via ``app.dependency_overrides[get_services]``.
    """
    return request.app.state.services
