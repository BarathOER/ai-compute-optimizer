"""Unit tests for cross-encoder score normalization.

``to_probabilities`` is the single source of truth shared by the production
reranker (``app/reranker.py``) and the offline eval (``eval/two_stage_eval.py``).
These tests pin its behaviour on all three output shapes a CrossEncoder can
return, and guard against the double-sigmoid regression that made live scores
wrong (a reworded match reading 0.705, an unrelated pair 0.500).
"""

from __future__ import annotations

import math

import numpy as np

from app.reranker import Reranker, to_probabilities


def test_two_column_logits_use_softmax_positive_class() -> None:
    """A 2-class logit matrix -> softmax, take the positive-class column."""
    raw = np.array([[2.0, -2.0], [0.0, 0.0], [-1.0, 3.0]])
    probs = to_probabilities(raw)

    assert probs.shape == (3,)
    assert np.all((probs > 0.0) & (probs < 1.0))
    # Row [0,0] is an even split.
    assert probs[1] == 0.5
    # Monotonic in (positive - negative) logit: row2 > row1 > row0.
    assert probs[2] > probs[1] > probs[0]


def test_one_dim_logits_get_sigmoid() -> None:
    """Raw 1-D logits outside [0, 1] are squashed with a sigmoid."""
    raw = np.array([-4.0, 0.0, 4.0])
    probs = to_probabilities(raw)

    assert probs[1] == 0.5
    assert probs[0] < 0.05 and probs[2] > 0.95
    assert probs[0] < probs[1] < probs[2]  # monotonic
    assert math.isclose(probs[2], 1.0 / (1.0 + math.exp(-4.0)))


def test_values_already_in_unit_range_pass_through() -> None:
    """Values already in [0, 1] must NOT be sigmoided again (the live bug).

    ``quora-distilroberta-base`` applies its own sigmoid inside predict(), so
    its output is already a probability. A second sigmoid is what made a strong
    match (~0.95) read as ~0.72 and nothing ever clear 0.943.
    """
    raw = np.array([0.02, 0.5, 0.95])
    probs = to_probabilities(raw)

    assert np.allclose(probs, raw)  # unchanged
    # A second sigmoid would have moved 0.95 to ~0.721 -- assert it did not.
    assert probs[2] == 0.95
    assert abs(probs[2] - 1.0 / (1.0 + math.exp(-0.95))) > 0.2


def test_all_shapes_preserve_ranking() -> None:
    """Whatever the shape, higher relevance stays higher after normalization."""
    two_col = to_probabilities(np.array([[3.0, -1.0], [1.0, 2.0]]))
    logits = to_probabilities(np.array([0.1, 5.0]))
    unit = to_probabilities(np.array([0.3, 0.8]))
    for probs in (two_col, logits, unit):
        assert probs[0] < probs[1]


def _reranker_with_model(model) -> Reranker:
    """Build a Reranker around ``model`` without loading a real cross-encoder."""
    r = Reranker.__new__(Reranker)
    r._model = model
    r._model_name = "stub"
    r._threshold = 0.9
    return r


def test_score_applies_shared_normalization() -> None:
    """Reranker.score routes raw output through to_probabilities (no 2nd sigmoid)."""

    class StubModel:
        def predict(self, pairs, convert_to_numpy=True):
            # Already-probabilities, like quora-distilroberta-base.
            return np.array([0.95, 0.02][: len(pairs)])

    reranker = _reranker_with_model(StubModel())
    scores = reranker.score([("a", "a"), ("a", "b")])

    assert scores == [0.95, 0.02]  # passed through, not double-sigmoided
    assert all(isinstance(s, float) for s in scores)


def test_score_empty_input_short_circuits() -> None:
    """Empty input returns [] without invoking the model."""

    class BoomModel:
        def predict(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("predict should not be called for empty input")

    reranker = _reranker_with_model(BoomModel())
    assert reranker.score([]) == []
