"""Semantic cache backed by ChromaDB with two-stage retrieval.

A cached answer may be reused only when a new prompt means the *same thing* as a
stored one. Deciding that with a single bi-encoder cosine threshold is
unreliable (see ``eval/``): the classes barely separate, so any threshold either
misses real duplicates or serves wrong answers. Instead this cache retrieves in
two stages:

    Stage 1 (recall): ask Chroma for the top-k nearest cached prompts and keep
        those with cosine similarity >= ``stage1_threshold`` (default 0.70,
        tuned for recall — it decides *candidates*, not hits).
    Stage 2 (precision): a cross-encoder reranks those candidates by reading the
        query and the cached prompt jointly; accept a hit only if the best
        reranker score clears ``reranker.threshold``.

Chroma returns cosine *distance* in ``[0, 2]``; similarity is ``1 - distance``.
Every lookup reports both the stage-1 similarity and the stage-2 reranker score
so that misses remain diagnosable.

The reranker can be disabled (``enable_reranker=False``) to fall back to
bi-encoder-only matching — the older architecture — for A/B comparison and tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

    from app.reranker import Reranker


@dataclass(frozen=True)
class CacheHit:
    """A successful semantic-cache lookup."""

    answer: str
    similarity: float
    model: str
    reranker_score: float | None = None


@dataclass(frozen=True)
class LookupResult:
    """Outcome of a cache lookup.

    ``hit`` is ``None`` on a miss. ``best_similarity`` is the stage-1 cosine of
    the nearest cached prompt, and ``best_reranker_score`` is the highest
    stage-2 score among candidates (``None`` when the cache is empty, no
    candidate cleared stage 1, or the reranker is disabled). Reporting both on
    every lookup keeps misses diagnosable — you can see which stage rejected.
    """

    hit: CacheHit | None
    best_similarity: float | None
    best_reranker_score: float | None = None


@dataclass(frozen=True)
class _Candidate:
    """One stage-1 nearest neighbour under consideration for reranking."""

    prompt: str
    answer: str
    model: str
    similarity: float


def _make_client(
    *,
    use_remote: bool,
    host: str,
    port: int,
    persist_dir: str,
) -> "ClientAPI":
    """Create either an HTTP or a local persistent Chroma client."""
    import chromadb

    if use_remote:
        return chromadb.HttpClient(host=host, port=port)
    return chromadb.PersistentClient(path=persist_dir)


class SemanticCache:
    """A two-stage (bi-encoder recall + cross-encoder rerank) semantic cache."""

    def __init__(
        self,
        *,
        collection_name: str,
        stage1_threshold: float,
        top_k: int = 5,
        enable_reranker: bool = True,
        reranker: "Reranker | None" = None,
        use_remote: bool = False,
        host: str = "localhost",
        port: int = 8000,
        persist_dir: str = "./chroma_data",
        client: "ClientAPI | None" = None,
    ) -> None:
        if enable_reranker and reranker is None:
            raise ValueError(
                "enable_reranker=True requires a reranker instance."
            )
        self._stage1_threshold = stage1_threshold
        self._top_k = top_k
        self._enable_reranker = enable_reranker
        self._reranker = reranker
        self._client: "ClientAPI" = client or _make_client(
            use_remote=use_remote,
            host=host,
            port=port,
            persist_dir=persist_dir,
        )
        # Cosine space so distances map cleanly to similarity.
        self._collection: "Collection" = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def stage1_threshold(self) -> float:
        """The stage-1 cosine similarity required to become a rerank candidate."""
        return self._stage1_threshold

    @staticmethod
    def _make_id(prompt: str) -> str:
        """Deterministic id so identical prompts overwrite rather than dupe."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def size(self) -> int:
        """Number of cached entries."""
        return int(self._collection.count())

    def _stage1_candidates(self, embedding: list[float]) -> list[_Candidate]:
        """Return the top-k nearest cached prompts that clear stage 1.

        Chroma returns neighbours in ascending distance (descending similarity),
        so the list is already ordered best-first.
        """
        result: dict[str, Any] = self._collection.query(
            query_embeddings=[embedding],
            n_results=self._top_k,
            include=["documents", "metadatas", "distances"],
        )
        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]

        candidates: list[_Candidate] = []
        for distance, answer, metadata in zip(distances, documents, metadatas):
            similarity = 1.0 - float(distance)
            if similarity < self._stage1_threshold:
                continue  # ordered best-first, but keep explicit for clarity
            meta = metadata or {}
            candidates.append(
                _Candidate(
                    prompt=str(meta.get("prompt", "")),
                    answer=answer,
                    model=str(meta.get("model", "unknown")),
                    similarity=similarity,
                )
            )
        return candidates

    def lookup(self, prompt: str, embedding: list[float]) -> LookupResult:
        """Two-stage lookup: bi-encoder recall filter, then cross-encoder rerank.

        Returns a hit only if a candidate clears both stages (or, with the
        reranker disabled, the stage-1 filter alone). Always reports the best
        stage-1 similarity and stage-2 score seen so misses are diagnosable.
        """
        if self._collection.count() == 0:
            return LookupResult(hit=None, best_similarity=None)

        # Peek at the single nearest neighbour so a stage-1 miss still reports
        # the similarity it rejected.
        nearest = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["distances"],
        )
        nearest_distances = (nearest.get("distances") or [[]])[0]
        best_similarity = (
            1.0 - float(nearest_distances[0]) if nearest_distances else None
        )

        candidates = self._stage1_candidates(embedding)
        if not candidates:
            return LookupResult(hit=None, best_similarity=best_similarity)

        # --- Stage 2 disabled: bi-encoder-only fallback (older architecture) --
        if not self._enable_reranker or self._reranker is None:
            top = candidates[0]
            hit = CacheHit(answer=top.answer, similarity=top.similarity, model=top.model)
            return LookupResult(hit=hit, best_similarity=best_similarity)

        # --- Stage 2: rerank candidates jointly with the query ---------------
        scores = self._reranker.score([(prompt, c.prompt) for c in candidates])
        best_idx = max(range(len(scores)), key=scores.__getitem__)
        best_reranker_score = scores[best_idx]

        if best_reranker_score < self._reranker.threshold:
            return LookupResult(
                hit=None,
                best_similarity=best_similarity,
                best_reranker_score=best_reranker_score,
            )

        chosen = candidates[best_idx]
        hit = CacheHit(
            answer=chosen.answer,
            similarity=chosen.similarity,
            model=chosen.model,
            reranker_score=best_reranker_score,
        )
        return LookupResult(
            hit=hit,
            best_similarity=best_similarity,
            best_reranker_score=best_reranker_score,
        )

    def store(
        self,
        prompt: str,
        embedding: list[float],
        answer: str,
        model: str,
    ) -> None:
        """Insert (or overwrite) a prompt/answer pair in the cache."""
        self._collection.upsert(
            ids=[self._make_id(prompt)],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[{"prompt": prompt, "model": model}],
        )
