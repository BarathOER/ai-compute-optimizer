"""End-to-end tests for the gateway: health, cache hit, cache miss, routing."""

from __future__ import annotations

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

    # Same semantic content -> cosine similarity above threshold -> hit.
    second = client.post("/query", json={"prompt": "capital of france"})
    body = second.json()
    assert body["cache_hit"] is True
    assert body["route"] == "cache"
    assert body["similarity"] >= 0.9
    assert body["estimated_cost_usd"] == 0.0
    assert body["estimated_savings_usd"] > 0.0

    # Crucially, no second LLM call happened.
    assert len(fake_llm.calls) == 1


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
