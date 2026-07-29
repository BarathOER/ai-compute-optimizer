"""Test fixtures.

We avoid downloading real embedding models and hitting real LLMs/ChromaDB by
injecting lightweight fakes through FastAPI's ``dependency_overrides``. The
fakes preserve the *contracts* the app relies on:

* a deterministic embedder (bag-of-words vector) so similar prompts land near
  each other in cosine space,
* an in-memory cache with the same lookup/store semantics,
* a scripted LLM client that records which route it was called with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.cache import CacheHit, LookupResult
from app.config import Settings
from app.llm import LLMResult
from app.main import app
from app.metrics import Metrics
from app.router import ComplexityRouter, Route
from app.services import Services, get_services


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Produces an L2-normalized bag-of-words vector over a small fixed vocabulary
    so cosine similarity between overlapping prompts is high.
    """

    _VOCAB = [
        "capital",
        "france",
        "paris",
        "weather",
        "today",
        "python",
        "list",
        "sort",
        "explain",
        "quantum",
    ]

    def __init__(self, model_name: str = "fake") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vector = [float(tokens.count(word)) for word in self._VOCAB]
        # Add a tiny constant so all-zero vectors are still valid/normalizable.
        vector = [v + 0.01 for v in vector]
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


@dataclass
class FakeReranker:
    """Deterministic stand-in for the cross-encoder.

    Scores a ``(query, candidate)`` pair by Jaccard token overlap: identical
    prompts score 1.0, near-variants score below the (high) threshold. This lets
    tests exercise stage-2 acceptance and rejection without loading a model.
    """

    threshold: float
    model_name: str = "fake-reranker"

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 1.0
        return len(ta & tb) / len(ta | tb)

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._jaccard(a, b) for a, b in pairs]


@dataclass
class _Entry:
    prompt: str
    embedding: list[float]
    answer: str
    model: str


@dataclass
class FakeCache:
    """In-memory two-stage cache mirroring SemanticCache semantics."""

    stage1_threshold: float
    enable_reranker: bool = True
    reranker: FakeReranker | None = None
    top_k: int = 5
    _entries: dict[str, _Entry] = field(default_factory=dict)

    def size(self) -> int:
        return len(self._entries)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def lookup(self, prompt: str, embedding: list[float]) -> LookupResult:
        """Mirror SemanticCache: stage-1 recall filter, then stage-2 rerank."""
        if not self._entries:
            return LookupResult(hit=None, best_similarity=None)

        scored = sorted(
            ((self._cosine(embedding, e.embedding), e) for e in self._entries.values()),
            key=lambda pair: pair[0],
            reverse=True,
        )
        best_similarity = scored[0][0]
        candidates = [(s, e) for s, e in scored if s >= self.stage1_threshold][
            : self.top_k
        ]
        if not candidates:
            return LookupResult(hit=None, best_similarity=best_similarity)

        # Stage 2 disabled: bi-encoder-only fallback accepts the top candidate.
        if not self.enable_reranker or self.reranker is None:
            sim, entry = candidates[0]
            hit = CacheHit(answer=entry.answer, similarity=sim, model=entry.model)
            return LookupResult(hit=hit, best_similarity=best_similarity)

        scores = self.reranker.score([(prompt, e.prompt) for _, e in candidates])
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        best_reranker_score = scores[best_idx]
        if best_reranker_score < self.reranker.threshold:
            return LookupResult(
                hit=None,
                best_similarity=best_similarity,
                best_reranker_score=best_reranker_score,
            )
        sim, entry = candidates[best_idx]
        hit = CacheHit(
            answer=entry.answer,
            similarity=sim,
            model=entry.model,
            reranker_score=best_reranker_score,
        )
        return LookupResult(
            hit=hit,
            best_similarity=best_similarity,
            best_reranker_score=best_reranker_score,
        )

    def store(
        self, prompt: str, embedding: list[float], answer: str, model: str
    ) -> None:
        self._entries[prompt] = _Entry(
            prompt=prompt, embedding=embedding, answer=answer, model=model
        )


@dataclass
class FakeLLM:
    """Scripted LLM that records the route it was asked to serve."""

    local_model: str = "fake-local"
    remote_model: str = "fake-remote"
    calls: list[Route] = field(default_factory=list)

    async def generate(self, prompt: str, route: Route) -> LLMResult:
        self.calls.append(route)
        model = self.remote_model if route == "remote" else self.local_model
        return LLMResult(text=f"[{route}] answer to: {prompt}", model=model)


@pytest.fixture
def settings() -> Settings:
    """Test settings: two-stage thresholds plus a small word threshold."""
    return Settings(
        stage1_threshold=0.7,
        rerank_top_k=5,
        enable_reranker=True,
        reranker_threshold=0.943,
        complexity_word_threshold=8,
        remote_input_cost_per_1m=1.50,
        remote_output_cost_per_1m=9.00,
        local_input_cost_per_1m=0.0,
        local_output_cost_per_1m=0.0,
        projected_monthly_queries=100_000,
        gemini_api_key="test-key",
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


def build_fake_services(
    settings: Settings,
    fake_llm: FakeLLM,
    *,
    enable_reranker: bool = True,
    enable_local_route: bool = True,
) -> Services:
    """Assemble a Services container from fakes, reranker on or off."""
    reranker = FakeReranker(threshold=settings.reranker_threshold) if enable_reranker else None
    return Services(
        settings=settings,
        embedder=FakeEmbedder(),
        cache=FakeCache(
            stage1_threshold=settings.stage1_threshold,
            enable_reranker=enable_reranker,
            reranker=reranker,
            top_k=settings.rerank_top_k,
        ),
        router=ComplexityRouter(
            settings.complexity_word_threshold,
            enable_local_route=enable_local_route,
        ),
        llm=fake_llm,
        metrics=Metrics(
            remote_input_cost_per_1m=settings.remote_input_cost_per_1m,
            remote_output_cost_per_1m=settings.remote_output_cost_per_1m,
            local_input_cost_per_1m=settings.local_input_cost_per_1m,
            local_output_cost_per_1m=settings.local_output_cost_per_1m,
            monthly_query_volume=settings.projected_monthly_queries,
        ),
        reranker=reranker,
    )


@pytest.fixture
def services(settings: Settings, fake_llm: FakeLLM) -> Services:
    """Two-stage Services (reranker enabled) built entirely from fakes."""
    return build_fake_services(settings, fake_llm, enable_reranker=True)


@pytest.fixture
def services_no_reranker(settings: Settings, fake_llm: FakeLLM) -> Services:
    """Bi-encoder-only Services (stage 2 bypassed) for the A/B fallback path."""
    return build_fake_services(settings, fake_llm, enable_reranker=False)


@pytest.fixture
def services_gemini_only(settings: Settings, fake_llm: FakeLLM) -> Services:
    """Cloud-style Services with the local route disabled (Gemini-only)."""
    return build_fake_services(settings, fake_llm, enable_local_route=False)


def _client_for(services: Services) -> TestClient:
    """A TestClient with the Services dependency overridden by ``services``.

    ``TestClient`` is used *without* its context-manager form so the app's
    lifespan (which would build real chromadb/sentence-transformers services)
    does not run. Handlers resolve ``get_services`` through the override.
    """
    app.dependency_overrides[get_services] = lambda: services
    app.state.services = services
    return TestClient(app)


@pytest.fixture
def client(services: Services):
    test_client = _client_for(services)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_reranker(services_no_reranker: Services):
    test_client = _client_for(services_no_reranker)
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def client_gemini_only(services_gemini_only: Services):
    test_client = _client_for(services_gemini_only)
    yield test_client
    app.dependency_overrides.clear()