"""FastAPI application: wiring, lifespan, and the public endpoints.

Request flow for ``/query``:

    embed(prompt)
      -> cache.lookup            # cosine >= threshold => hit
        hit  -> return cached answer (cost 0)
        miss -> router.route     # local (cheap) vs remote (frontier)
             -> llm.generate
             -> cache.store
             -> return answer
    ... every request is timed and recorded in metrics. The nearest-neighbour
    similarity is reported on misses too, so the threshold can be tuned.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException

from app import __version__
from app.config import Settings, get_settings
from app.metrics import estimate_tokens
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.services import Services, build_services, get_services

logger = logging.getLogger("ai_compute_optimizer")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build real Services on startup and stash them on ``app.state``."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger.info("Starting %s v%s", settings.app_name, __version__)
    app.state.services = build_services(settings)
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title="AI Compute Optimizer",
    version=__version__,
    description="An LLM cost-reduction API gateway with semantic caching and "
    "complexity-based routing.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(services: Services = Depends(get_services)) -> HealthResponse:
    """Liveness probe plus current cache size."""
    return HealthResponse(
        app=services.settings.app_name,
        version=__version__,
        cache_size=services.cache.size(),
    )


@app.get("/metrics", tags=["ops"])
async def metrics(services: Services = Depends(get_services)) -> dict:
    """Return the current metrics snapshot (hit rate, latency, cost, savings)."""
    return services.metrics.as_dict()


@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(
    payload: QueryRequest,
    services: Services = Depends(get_services),
) -> QueryResponse:
    """Answer a prompt from cache when possible, otherwise route to an LLM."""
    start = time.perf_counter()
    embedding = services.embedder.embed(payload.prompt)

    # --- Cache lookup ----------------------------------------------------
    # Keep the nearest-neighbour score even on a miss so it lands in the
    # response; you cannot tune a threshold you cannot measure.
    best_similarity: float | None = None
    if not payload.force_refresh:
        lookup = services.cache.lookup(payload.prompt, embedding)
        best_similarity = lookup.best_similarity
        if lookup.hit is not None:
            hit = lookup.hit
            latency_ms = (time.perf_counter() - start) * 1000.0
            actual_cost, savings = services.metrics.record(
                cache_hit=True,
                route="cache",
                latency_ms=latency_ms,
                input_tokens=estimate_tokens(payload.prompt),
                output_tokens=estimate_tokens(hit.answer),
            )
            return QueryResponse(
                answer=hit.answer,
                cache_hit=True,
                route="cache",
                model=hit.model,
                similarity=hit.similarity,
                latency_ms=latency_ms,
                estimated_cost_usd=actual_cost,
                estimated_savings_usd=savings,
            )

    # --- Cache miss: route and call an LLM -------------------------------
    route = services.router.route(payload.prompt)
    try:
        result = await services.llm.generate(payload.prompt, route)
    except Exception as exc:  # surface backend failures as 502s
        logger.exception("LLM backend failed on route=%s", route)
        raise HTTPException(status_code=502, detail=f"LLM backend error: {exc}")

    services.cache.store(payload.prompt, embedding, result.text, result.model)

    latency_ms = (time.perf_counter() - start) * 1000.0
    actual_cost, savings = services.metrics.record(
        cache_hit=False,
        route=route,
        latency_ms=latency_ms,
        input_tokens=estimate_tokens(payload.prompt),
        output_tokens=estimate_tokens(result.text),
    )
    return QueryResponse(
        answer=result.text,
        cache_hit=False,
        route=route,
        model=result.model,
        # The rejected nearest-neighbour score — this is the tuning signal.
        similarity=best_similarity,
        latency_ms=latency_ms,
        estimated_cost_usd=actual_cost,
        estimated_savings_usd=savings,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Return the FastAPI app (hook for advanced/testing scenarios)."""
    if settings is not None:
        app.state.services = build_services(settings)
    return app