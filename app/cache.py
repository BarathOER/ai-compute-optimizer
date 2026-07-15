"""Semantic cache backed by ChromaDB.

Prompts are stored with their embedding vectors. A lookup embeds the incoming
prompt and asks Chroma for the nearest neighbour by cosine distance; if the
resulting cosine *similarity* meets the configured threshold, it is a hit.

Chroma returns cosine *distance* in ``[0, 2]``; similarity is ``1 - distance``.

Every lookup reports the best similarity it saw — hit or miss — so the
threshold can be tuned against real data rather than guessed at.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


@dataclass(frozen=True)
class CacheHit:
    """A successful semantic-cache lookup."""

    answer: str
    similarity: float
    model: str


@dataclass(frozen=True)
class LookupResult:
    """Outcome of a cache lookup.

    ``hit`` is ``None`` on a miss. ``best_similarity`` is the score of the
    nearest cached prompt regardless of outcome (``None`` only when the cache
    is empty or Chroma returned nothing), which is what makes threshold
    tuning measurable.
    """

    hit: CacheHit | None
    best_similarity: float | None


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
    """A cosine-similarity semantic cache over prompt embeddings."""

    def __init__(
        self,
        *,
        collection_name: str,
        similarity_threshold: float,
        use_remote: bool = False,
        host: str = "localhost",
        port: int = 8000,
        persist_dir: str = "./chroma_data",
        client: "ClientAPI | None" = None,
    ) -> None:
        self._threshold = similarity_threshold
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
    def threshold(self) -> float:
        """The configured minimum cosine similarity for a hit."""
        return self._threshold

    @staticmethod
    def _make_id(prompt: str) -> str:
        """Deterministic id so identical prompts overwrite rather than dupe."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def size(self) -> int:
        """Number of cached entries."""
        return int(self._collection.count())

    def lookup(self, prompt: str, embedding: list[float]) -> LookupResult:
        """Return the nearest cached match and its similarity, hit or miss."""
        if self._collection.count() == 0:
            return LookupResult(hit=None, best_similarity=None)

        result: dict[str, Any] = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        distances = result.get("distances") or [[]]
        if not distances or not distances[0]:
            return LookupResult(hit=None, best_similarity=None)

        distance = float(distances[0][0])
        similarity = 1.0 - distance

        # Miss: no hit, but report the score we rejected so it can be tuned.
        if similarity < self._threshold:
            return LookupResult(hit=None, best_similarity=similarity)

        documents = result.get("documents") or [[""]]
        metadatas = result.get("metadatas") or [[{}]]
        metadata = metadatas[0][0] or {}
        hit = CacheHit(
            answer=documents[0][0],
            similarity=similarity,
            model=str(metadata.get("model", "unknown")),
        )
        return LookupResult(hit=hit, best_similarity=similarity)

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