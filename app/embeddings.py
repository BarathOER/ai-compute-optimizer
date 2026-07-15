"""Prompt embedding via sentence-transformers.

The model is loaded lazily on first use so that importing this module (e.g. in
tests) does not trigger a multi-hundred-MB download.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sentence_transformers import SentenceTransformer


class Embedder:
    """Wraps a sentence-transformers model to produce prompt embeddings."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: "SentenceTransformer | None" = None

    @property
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        return self._model_name

    def _ensure_model(self) -> "SentenceTransformer":
        """Load the model on first use and cache it for subsequent calls."""
        if self._model is None:
            # Imported here (not at module top) to keep the dependency lazy.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for a single string."""
        model = self._ensure_model()
        vector = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of strings."""
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors.tolist()]
