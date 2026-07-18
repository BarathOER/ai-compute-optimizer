"""Cross-encoder reranker for stage 2 of semantic-cache retrieval.

A bi-encoder embeds each prompt independently, which is fast but blind to the
fine distinctions that flip meaning ("how do I *learn* X" vs. "how do I *teach*
X"). Offline evaluation (see ``eval/``) showed a bi-encoder alone reaches only
~7.7% recall at 95% precision on QQP, and that this is architectural: three
different bi-encoders hit the same wall. A cross-encoder reads *both* prompts
jointly and lifts recall to ~68.8% at 95% precision for ~+12 ms per query.

This module wraps that cross-encoder. It is loaded once at startup (see
``app.services.build_services``) and reused for every request.

IMPORTANT: cross-encoder scores are NOT calibrated probabilities. The default
``RERANKER_THRESHOLD`` of 0.943 is an *empirical* cut point found for this
specific model (``cross-encoder/quora-distilroberta-base``) on QQP — it is a
decision boundary, not a confidence value, and it does not transfer to other
rerankers. Re-tune it with ``eval/two_stage_eval.py`` if the model changes.

SCORE SCALE: a CrossEncoder's ``predict()`` can return one of three shapes, and
applying the wrong transform silently corrupts the score. :func:`to_probabilities`
handles all three (2-column logits, raw 1-D logits, or values already in [0, 1])
and is the SINGLE source of truth shared with ``eval/two_stage_eval.py`` so the
served scores can never drift from the ones the threshold was tuned against.
``cross-encoder/quora-distilroberta-base`` in particular already applies its own
sigmoid inside ``predict()`` — its output is already in [0, 1], so it must be
used as-is. (Sigmoiding it a second time squashes a strong match's ~0.95 to
~0.72 and an unrelated pair's ~0.0 to 0.5, so nothing ever clears 0.943.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def to_probabilities(raw: "np.ndarray | list") -> np.ndarray:
    """Map raw cross-encoder ``predict()`` output to (0, 1), monotonically.

    Handles the three shapes a CrossEncoder can return; thresholding only needs
    monotonicity, which all three preserve:

    * **2-column logit matrix** (a 2-class head) -> softmax, take the
      positive-class column.
    * **1-D logits outside [0, 1]** (a regression head without an output
      activation) -> sigmoid.
    * **values already in [0, 1]** (the head already applied a sigmoid, as
      ``quora-distilroberta-base`` does) -> used as-is; do NOT sigmoid again.
    """
    scores = np.asarray(raw, dtype=float)
    if scores.ndim == 2 and scores.shape[1] == 2:
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return (exp / exp.sum(axis=1, keepdims=True))[:, 1]
    scores = scores.ravel()
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        return 1.0 / (1.0 + np.exp(-scores))
    return scores


def _describe_shape(raw: "np.ndarray | list") -> str:
    """Human-readable label for which :func:`to_probabilities` branch applies."""
    scores = np.asarray(raw, dtype=float)
    if scores.ndim == 2 and scores.shape[1] == 2:
        return "2-column logits -> softmax(positive class)"
    flat = scores.ravel()
    if flat.size and (flat.min() < 0.0 or flat.max() > 1.0):
        return "1-D logits outside [0,1] -> sigmoid"
    return "already in [0,1] -> used as-is (no sigmoid)"


class Reranker:
    """Scores (query, candidate) prompt pairs with a cross-encoder."""

    def __init__(self, model_name: str, threshold: float) -> None:
        self._model_name = model_name
        self._threshold = threshold
        # Load eagerly: constructing a Reranker downloads/initializes the model,
        # which we want to happen once at application startup, never per request.
        from sentence_transformers import CrossEncoder

        self._model: "CrossEncoder" = CrossEncoder(model_name)
        self._log_output_shape()

    @property
    def model_name(self) -> str:
        """Name of the underlying cross-encoder model."""
        return self._model_name

    @property
    def threshold(self) -> float:
        """Empirical accept threshold (not a probability; model-specific)."""
        return self._threshold

    def _log_output_shape(self) -> None:
        """Probe the model once at startup and log which normalization applies.

        Emitted at DEBUG (set ``LOG_LEVEL=DEBUG``) so the actual raw output shape
        of the configured reranker can be confirmed against
        :func:`to_probabilities` without guessing.
        """
        try:
            raw = self._model.predict(
                [("probe query", "probe candidate")], convert_to_numpy=True
            )
            arr = np.asarray(raw, dtype=float)
            logger.debug(
                "Reranker '%s' predict() raw output: shape=%s ndim=%d "
                "min=%.4f max=%.4f -> %s",
                self._model_name, arr.shape, arr.ndim,
                float(arr.min()), float(arr.max()), _describe_shape(raw),
            )
        except Exception:  # a diagnostic probe must never break startup
            logger.debug(
                "Reranker '%s' output-shape probe failed", self._model_name,
                exc_info=True,
            )

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one relevance score in (0, 1) per ``(query, candidate)`` pair.

        Higher means "more likely the same question". Raw model output is
        normalized via :func:`to_probabilities` (the same transform the eval
        uses). Compare against :attr:`threshold` to decide acceptance. Returns
        an empty list for empty input so callers can pass through cheaply.
        """
        if not pairs:
            return []
        raw = self._model.predict(list(pairs), convert_to_numpy=True)
        return [float(score) for score in to_probabilities(raw)]
