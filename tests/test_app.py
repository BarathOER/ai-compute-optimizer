"""End-to-end tests for the gateway: health, cache hit, cache miss, routing."""

from __future__ import annotations

from app.metrics import Metrics
from app.router import ComplexityRouter


def test_health_ok(client) -> None:
    """/health returns ok and reports an (initially empty) cache."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["cache_size"] == 0


def test_cache_miss_calls_llm_and_stores(client, fake_llm) -> None:
    """A first-time prompt misses the cache, calls the LLM, and is stored."""
    response = client.post("/query", json={"prompt": "capital of france"})
    assert response.status_code == 200
    body = response.json()

    assert body["cache_hit"] is False
    assert body["route"] in {"local", "remote"}
    assert body["answer"].endswith("capital of france")
    assert len(fake_llm.calls) == 1  # the LLM was actually invoked

    # It was stored, so the cache now has one entry.
    assert client.get("/health").json()["cache_size"] == 1


def test_cache_hit_returns_cached_answer_without_llm(client, fake_llm) -> None:
    """A near-identical follow-up prompt is served from cache, no LLM call."""
    first = client.post("/query", json={"prompt": "capital of france"})
    assert first.json()["cache_hit"] is False
    assert len(fake_llm.calls) == 1

    # Same content -> stage-1 cosine high AND stage-2 reranker clears -> hit.
    second = client.post("/query", json={"prompt": "capital of france"})
    body = second.json()
    assert body["cache_hit"] is True
    assert body["route"] == "cache"
    assert body["similarity"] >= 0.9
    # Stage-2 ran and accepted: the reranker score is reported on the hit.
    assert body["reranker_score"] is not None
    assert body["reranker_score"] >= 0.943
    assert body["estimated_cost_usd"] == 0.0
    assert body["estimated_savings_usd"] > 0.0

    # Crucially, no second LLM call happened.
    assert len(fake_llm.calls) == 1


def test_two_stage_reranker_rejects_stage1_match(client, fake_llm) -> None:
    """A prompt that clears stage 1 but not the reranker is a miss (diagnosable)."""
    client.post("/query", json={"prompt": "capital of france"})
    assert len(fake_llm.calls) == 1

    # "capital of france please" embeds identically (extra word is out of the
    # fake vocab), so stage-1 cosine passes -- but the reranker scores the
    # non-identical pair below threshold and rejects it.
    response = client.post("/query", json={"prompt": "capital of france please"})
    body = response.json()
    assert body["cache_hit"] is False
    assert body["similarity"] >= 0.9  # stage 1 did match
    assert body["reranker_score"] is not None
    assert body["reranker_score"] < 0.943  # but stage 2 rejected it
    # The reranker miss fell through to the LLM.
    assert len(fake_llm.calls) == 2


def test_reranker_disabled_accepts_stage1_match(client_no_reranker, fake_llm) -> None:
    """With the reranker off, a stage-1 match alone is a hit (bi-encoder-only)."""
    client_no_reranker.post("/query", json={"prompt": "capital of france"})
    assert len(fake_llm.calls) == 1

    # Same near-variant as above: rejected in two-stage mode, accepted here.
    response = client_no_reranker.post(
        "/query", json={"prompt": "capital of france please"}
    )
    body = response.json()
    assert body["cache_hit"] is True
    assert body["route"] == "cache"
    assert body["reranker_score"] is None  # stage 2 never ran
    assert len(fake_llm.calls) == 1  # served from cache, no new LLM call


def test_force_refresh_bypasses_cache(client, fake_llm) -> None:
    """force_refresh always calls the LLM even when a cache entry exists."""
    client.post("/query", json={"prompt": "capital of france"})
    assert len(fake_llm.calls) == 1

    client.post(
        "/query", json={"prompt": "capital of france", "force_refresh": True}
    )
    assert len(fake_llm.calls) == 2


def test_router_simple_prompt_goes_local() -> None:
    """A short, plain prompt routes to the local model."""
    router = ComplexityRouter(word_threshold=40)
    assert router.route("what is 2 plus 2") == "local"
    assert router.is_complex("what is 2 plus 2") is False


def test_router_complex_prompt_goes_remote() -> None:
    """Keyword-, length-, and multi-question-based complexity route remote."""
    router = ComplexityRouter(word_threshold=8)

    # Keyword trigger.
    assert router.route("analyze the tradeoffs here") == "remote"
    # Length trigger.
    long_prompt = " ".join(["word"] * 20)
    assert router.route(long_prompt) == "remote"
    # Multiple questions.
    assert router.route("who? what? why?") == "remote"


def test_routing_reflected_in_query_response(client, fake_llm) -> None:
    """The route chosen by the router is reflected in the API response."""
    response = client.post(
        "/query", json={"prompt": "design a scalable architecture please"}
    )
    body = response.json()
    assert body["route"] == "remote"
    assert fake_llm.calls == ["remote"]


def test_router_local_disabled_forces_remote() -> None:
    """With the local route disabled, even a simple prompt routes remote."""
    router = ComplexityRouter(word_threshold=40, enable_local_route=False)
    assert router.route("what is 2 plus 2") == "remote"   # simple, but no local
    assert router.route("analyze the tradeoffs here") == "remote"
    # The complexity heuristic itself is unchanged.
    assert router.is_complex("what is 2 plus 2") is False


def test_gemini_only_miss_routes_remote_then_cache_hits(
    client_gemini_only, fake_llm
) -> None:
    """Cloud mode (no Ollama): a miss goes to Gemini; a repeat still hits cache.

    Mirrors ENABLE_LOCAL_ROUTE=false with no local backend available — the
    two-stage cache is unchanged, only miss-routing changes.
    """
    # Simple prompt that would normally route local -> must go remote here.
    first = client_gemini_only.post("/query", json={"prompt": "capital of france"})
    body = first.json()
    assert body["cache_hit"] is False
    assert body["route"] == "remote"            # not "local"
    assert fake_llm.calls == ["remote"]         # the LLM was hit remotely
    assert body["answer"].endswith("capital of france")

    # The cache still works: an identical repeat is served without any LLM call.
    second = client_gemini_only.post("/query", json={"prompt": "capital of france"})
    hit = second.json()
    assert hit["cache_hit"] is True
    assert hit["route"] == "cache"
    assert fake_llm.calls == ["remote"]         # still just the one remote call


def test_metrics_endpoint_tracks_hits_and_savings(client) -> None:
    """/metrics reflects requests, hit rate, and accumulated savings."""
    client.post("/query", json={"prompt": "capital of france"})  # miss
    client.post("/query", json={"prompt": "capital of france"})  # hit

    snapshot = client.get("/metrics").json()
    assert snapshot["total_requests"] == 2
    assert snapshot["cache_hits"] == 1
    assert snapshot["cache_misses"] == 1
    assert snapshot["hit_rate"] == 0.5
    assert snapshot["total_savings_usd"] > 0.0
    assert snapshot["total_input_tokens"] > 0
    assert snapshot["total_output_tokens"] > 0


def _metrics() -> Metrics:
    """A Metrics with the split-rate pricing used across cost tests."""
    return Metrics(
        remote_input_cost_per_1m=1.50,
        remote_output_cost_per_1m=9.00,
        local_input_cost_per_1m=0.0,
        local_output_cost_per_1m=0.0,
        monthly_query_volume=100_000,
    )


def test_output_tokens_cost_six_times_input() -> None:
    """Output tokens are priced 6x input (9.00 vs 1.50 per 1M)."""
    metrics = _metrics()
    input_only = metrics.cost_for(1_000_000, 0, remote=True)
    output_only = metrics.cost_for(0, 1_000_000, remote=True)
    assert input_only == 1.50
    assert output_only == 9.00
    assert output_only == 6 * input_only


def test_cache_hit_saves_full_remote_cost() -> None:
    """A cache hit costs nothing and saves the whole remote price."""
    metrics = _metrics()
    actual, savings = metrics.record(
        cache_hit=True,
        route="cache",
        latency_ms=1.0,
        input_tokens=500,
        output_tokens=1_000,
    )
    expected_remote = metrics.cost_for(500, 1_000, remote=True)
    assert actual == 0.0
    assert savings == expected_remote > 0.0


def test_local_route_saves_difference_vs_remote() -> None:
    """A local-routed miss saves the remote price minus the (free) local cost."""
    metrics = _metrics()
    actual, savings = metrics.record(
        cache_hit=False,
        route="local",
        latency_ms=1.0,
        input_tokens=800,
        output_tokens=400,
    )
    assert actual == 0.0  # local rates are zero here
    assert savings == metrics.cost_for(800, 400, remote=True) > 0.0


def test_remote_route_has_no_savings() -> None:
    """A remote-routed miss pays full price and saves nothing."""
    metrics = _metrics()
    actual, savings = metrics.record(
        cache_hit=False,
        route="remote",
        latency_ms=1.0,
        input_tokens=800,
        output_tokens=400,
    )
    assert actual == metrics.cost_for(800, 400, remote=True)
    assert savings == 0.0


def test_projection_scales_with_volume_and_hit_rate() -> None:
    """Projected monthly/annual savings follow volume * hit_rate * avg cost."""
    metrics = _metrics()
    # One remote miss, one hit -> hit_rate 0.5, uniform token counts.
    metrics.record(
        cache_hit=False, route="remote", latency_ms=1.0,
        input_tokens=1_000, output_tokens=2_000,
    )
    metrics.record(
        cache_hit=True, route="cache", latency_ms=1.0,
        input_tokens=1_000, output_tokens=2_000,
    )

    projection = metrics.snapshot().projection
    per_query = metrics.cost_for(1_000, 2_000, remote=True)
    expected_monthly = 100_000 * 0.5 * per_query
    assert projection.hit_rate == 0.5
    assert projection.avg_input_tokens == 1_000
    assert projection.avg_output_tokens == 2_000
    assert projection.projected_monthly_savings_usd == expected_monthly
    assert projection.projected_annual_savings_usd == expected_monthly * 12


def test_metrics_endpoint_exposes_projection(client) -> None:
    """The /metrics payload includes the nested savings projection."""
    client.post("/query", json={"prompt": "capital of france"})  # miss
    client.post("/query", json={"prompt": "capital of france"})  # hit

    projection = client.get("/metrics").json()["projection"]
    assert projection["monthly_query_volume"] == 100_000
    assert projection["projected_monthly_savings_usd"] > 0.0
    assert (
        projection["projected_annual_savings_usd"]
        == projection["projected_monthly_savings_usd"] * 12
    )

    # The projection must carry a provenance "basis" so callers never read it
    # as fact: volume is assumed, hit rate / tokens / prices are measured.
    basis = projection["basis"]
    assert basis["monthly_query_volume"].startswith("ASSUMED")
    assert basis["hit_rate"].startswith("MEASURED")
    assert basis["token_prices"].startswith("MEASURED")
    assert basis["local_route_cost"].startswith("ASSUMED")
