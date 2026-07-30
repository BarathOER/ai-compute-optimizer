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

# NOTE: deliberately NOT using `from __future__ import annotations`.
# slowapi's @limiter.limit wraps the endpoint, and its wrapper carries slowapi's
# module globals. Under PEP 563 (stringized annotations), FastAPI would try to
# resolve the endpoint's string annotations (e.g. "QueryRequest") against those
# wrong globals, fail silently, and misclassify the Pydantic body as a query
# param -> 422. Real annotation objects need no such resolution.

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import __version__
from app.config import Settings, get_settings
from app.metrics import estimate_tokens
from app.models import HealthResponse, QueryRequest, QueryResponse
from app.services import Services, build_services, get_services

logger = logging.getLogger("ai_compute_optimizer")


def _rate_limit() -> str:
    """The active per-IP limit for /query, read from config (env RATE_LIMIT)."""
    return get_settings().rate_limit


def client_ip_key(request: Request) -> str:
    """Rate-limit key: the real client IP, honoring ``X-Forwarded-For``.

    On a platform like Render the app sits behind a proxy, so the socket peer
    (``request.client.host``) is the proxy — keying on it would make the limit
    *global* across all users. Proxies put the originating client IP as the
    first entry of ``X-Forwarded-For`` (``client, proxy1, proxy2``), so we use
    that. When there is no proxy header (local/Docker), fall back to the socket
    peer via slowapi's ``get_remote_address``.

    Note: ``X-Forwarded-For`` is client-settable when requests can reach the app
    directly, so this is sound for abuse/cost protection but not a hard security
    boundary. Behind a trusted proxy that overwrites the header, it's reliable.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client = forwarded.split(",")[0].strip()
        if client:
            return client
    return get_remote_address(request)


# Per-IP limiter. Only /query is decorated; /health and /metrics stay unlimited.
# headers_enabled=True so 429s carry Retry-After / RateLimit-* headers.
limiter = Limiter(key_func=client_ip_key, headers_enabled=True)


async def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a clean JSON 429 (with Retry-After / RateLimit-* headers)."""
    response = JSONResponse(
        status_code=429,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}. "
            "Please slow down and retry shortly."
        },
    )
    # slowapi attaches the standard rate-limit headers (Retry-After, etc.).
    return request.app.state.limiter._inject_headers(
        response, request.state.view_rate_limit
    )


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

# Wire the rate limiter: slowapi finds it on app.state.limiter, and the handler
# turns a tripped limit into a clean JSON 429.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
@limiter.limit(_rate_limit)
async def query(
    request: Request,
    response: Response,
    payload: QueryRequest,
    services: Services = Depends(get_services),
) -> QueryResponse:
    """Answer a prompt from cache when possible, otherwise route to an LLM.

    Rate-limited per client IP (``RATE_LIMIT``, default 10/minute). ``request``
    lets slowapi identify the caller; ``response`` lets it attach RateLimit-*
    headers to successful replies.
    """
    start = time.perf_counter()
    embedding = services.embedder.embed(payload.prompt)

    # --- Cache lookup (two-stage) ----------------------------------------
    # Keep both stage scores even on a miss so they land in the response;
    # you cannot diagnose or tune a threshold you cannot see.
    best_similarity: float | None = None
    best_reranker_score: float | None = None
    if not payload.force_refresh:
        lookup = services.cache.lookup(payload.prompt, embedding)
        best_similarity = lookup.best_similarity
        best_reranker_score = lookup.best_reranker_score
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
                reranker_score=hit.reranker_score,
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
        # The rejected stage scores — the signal for tuning either threshold.
        similarity=best_similarity,
        reranker_score=best_reranker_score,
        latency_ms=latency_ms,
        estimated_cost_usd=actual_cost,
        estimated_savings_usd=savings,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Return the FastAPI app (hook for advanced/testing scenarios)."""
    if settings is not None:
        app.state.services = build_services(settings)
    return app