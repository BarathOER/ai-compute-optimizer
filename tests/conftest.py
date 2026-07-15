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
class _Entry:
    embedding: list[float]
    answer: str
    model: str


@dataclass
class FakeCache:
    """In-memory cosine cache mirroring SemanticCache semantics."""

    similarity_threshold: float
    _entries: dict[str, _Entry] = field(default_factory=dict)

    @property
    def threshold(self) -> float:
        return self.similarity_threshold

    def size(self) -> int:
        return len(self._entries)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    def lookup(self, prompt: str, embedding: list[float]) -> LookupResult:
        """Mirror SemanticCache: always report the best similarity seen."""
        best: tuple[float, _Entry] | None = None
        for entry in self._entries.values():
            sim = self._cosine(embedding, entry.embedding)
            if best is None or sim > best[0]:
                best = (sim, entry)

        # Empty cache: nothing to compare against.
        if best is None:
            return LookupResult(hit=None, best_similarity=None)

        # Miss: no hit, but surface the rejected score for threshold tuning.
        if best[0] < self.similarity_threshold:
            return LookupResult(hit=None, best_similarity=best[0])

        hit = CacheHit(answer=best[1].answer, similarity=best[0], model=best[1].model)
        return LookupResult(hit=hit, best_similarity=best[0])

    def store(
        self, prompt: str, embedding: list[float], answer: str, model: str
    ) -> None:
        self._entries[prompt] = _Entry(embedding=embedding, answer=answer, model=model)


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
    """Test settings with a low-ish threshold and a small word threshold."""
    return Settings(
        similarity_threshold=0.9,
        complexity_word_threshold=8,
        remote_cost_per_1k_tokens=0.0005,
        local_cost_per_1k_tokens=0.0,
        gemini_api_key="test-key",
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings: Settings, fake_llm: FakeLLM) -> Services:
    """A Services container built entirely from fakes."""
    return Services(
        settings=settings,
        embedder=FakeEmbedder(),
        cache=FakeCache(similarity_threshold=settings.similarity_threshold),
        router=ComplexityRouter(settings.complexity_word_threshold),
        llm=fake_llm,
        metrics=Metrics(
            remote_cost_per_1k=settings.remote_cost_per_1k_tokens,
            local_cost_per_1k=settings.local_cost_per_1k_tokens,
        ),
    )


@pytest.fixture
def client(services: Services):
    """A TestClient whose Services dependency is overridden with fakes.

    ``TestClient`` is used *without* its context-manager form so the app's
    lifespan (which would build real chromadb/sentence-transformers services)
    does not run. Handlers resolve ``get_services`` through the override below.
    """
    app.dependency_overrides[get_services] = lambda: services
    app.state.services = services
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()