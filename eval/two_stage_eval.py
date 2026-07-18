"""Two-stage retrieval evaluation: bi-encoder recall filter + cross-encoder rerank.

Motivation. On QQP, independent bi-encoders (sentence embeddings compared by
cosine) hit an architectural wall: encoding each question alone discards the
fine distinctions that flip meaning ("How do I learn X?" vs. "How do I teach
X?"), so none reach 95% precision at usable recall and the class-separation gap
is ~0. A cross-encoder reads *both* questions jointly and can recover those
distinctions - at a latency cost. This script measures whether the two-stage
design is worth it, honestly including the latency it adds.

Pipeline
    Stage 1 (bi-encoder): cosine similarity; keep pairs >= --recall-threshold.
        Tuned for RECALL, not precision. Whatever true duplicates fall below
        this cut are lost forever - that fraction is the pipeline's recall
        CEILING; stage 2 can only discard, never recover.
    Stage 2 (cross-encoder): rerank the survivors by jointly reading both
        questions; sweep the reranker threshold.

All metrics are computed against the FULL sample (stage-1 drops count as false
negatives), so the numbers are directly comparable to the single-stage runs.

Run::

    python eval/two_stage_eval.py --n 5000 --dataset qqp \\
        --recall-threshold 0.70 \\
        --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --cross-encoder cross-encoder/quora-distilroberta-base

Evaluation only: no LLM calls, no server, and nothing under app/ is modified.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

# --- Make the project package importable when run as a script ------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.embeddings import Embedder  # noqa: E402
from app.reranker import to_probabilities  # noqa: E402  (single source of truth)
from eval.threshold_eval import (  # noqa: E402
    DEFAULT_OUT,
    THRESHOLDS,
    best_f1_row,
    compute_similarities,
    default_model,
    load_pairs,
    model_slug,
    precision_target_row,
    sweep_thresholds,
)

DEFAULT_CROSS_ENCODERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/quora-distilroberta-base",  # trained on this exact task
]


# ---------------------------------------------------------------------------
# Stage 1 - bi-encoder recall filter
# ---------------------------------------------------------------------------
def survivors_mask(sims: np.ndarray, recall_threshold: float) -> np.ndarray:
    """Boolean mask of pairs that clear the stage-1 recall filter."""
    return sims >= recall_threshold


def recall_ceiling(labels: np.ndarray, survivors: np.ndarray) -> float:
    """Fraction of true duplicates that survive stage 1 - the pipeline maximum.

    Stage 2 only removes candidates, so the two-stage recall can never exceed
    this. If it is low, no reranker can save the pipeline.
    """
    actual_pos = int(np.sum(labels == 1))
    if actual_pos == 0:
        return 0.0
    survived_pos = int(np.sum((labels == 1) & survivors))
    return survived_pos / actual_pos


# ---------------------------------------------------------------------------
# Stage 2 - cross-encoder rerank
# ---------------------------------------------------------------------------
def load_cross_encoder(model_name: str):
    """Load a sentence-transformers CrossEncoder (heavy import kept local)."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def score_pairs(model, pairs: list[tuple[str, str]]) -> np.ndarray:
    """Score ``pairs`` with an already-loaded cross-encoder -> probabilities.

    Uses the shared :func:`app.reranker.to_probabilities` so the eval scores
    match what the production reranker serves.
    """
    if not pairs:
        return np.asarray([], dtype=float)
    raw = model.predict(pairs, convert_to_numpy=True, show_progress_bar=True)
    return to_probabilities(raw)


def rerank(model_name: str, pairs: list[tuple[str, str]]):
    """Load a cross-encoder and score ``pairs``; return ``(probabilities, model)``.

    The model is returned so latency measurement can reuse the loaded instance.
    """
    model = load_cross_encoder(model_name)
    return score_pairs(model, pairs), model


def quantile_thresholds(probs: np.ndarray, count: int = 50) -> list[float]:
    """A sweep grid drawn from the score distribution itself.

    Cross-encoder scores are not uniform in [0, 1]; sweeping fixed steps would
    waste most of the grid. Sampling quantiles puts the cut points where scores
    actually are, tracing a clean PR curve. Rounded to 4 dp (see ``round_ndigits``).
    """
    if probs.size == 0:
        return [0.5]
    qs = np.quantile(probs, np.linspace(0.0, 1.0, count))
    return sorted({round(float(t), 4) for t in qs})


def two_stage_sweep(
    labels: np.ndarray,
    survivors: np.ndarray,
    survivor_probs: np.ndarray,
) -> pd.DataFrame:
    """Sweep the reranker threshold, scoring against the FULL sample.

    Non-survivors are parked below every threshold (score -1), so they can never
    be predicted positive and correctly count as false negatives when they are
    true duplicates. This reuses the single-stage sweep math for identical columns.
    """
    full_scores = np.full(labels.shape, -1.0, dtype=float)
    full_scores[survivors] = survivor_probs
    thresholds = quantile_thresholds(survivor_probs)
    return sweep_thresholds(full_scores, labels, thresholds, round_ndigits=4)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------
def _mean_p95(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.percentile(values, 95))


def measure_latency(
    embedder: Embedder,
    q_a: list[str],
    q_b: list[str],
    indices: np.ndarray,
    recall_threshold: float,
    cross_model=None,
) -> tuple[float, float]:
    """Per-pair latency (ms): stage 1 alone, or stage 1 + stage 2.

    When ``cross_model`` is given, the reranker is timed only for pairs that
    clear stage 1 - exactly as the pipeline runs in production - so the average
    reflects real cost, not a worst case. Excludes one-time model load via a
    warm-up call.
    """
    # Warm up so model load / lazy init is not charged to the first sample.
    _ = embedder.embed(q_a[indices[0]])
    _ = embedder.embed(q_b[indices[0]])
    if cross_model is not None:
        _ = cross_model.predict([(q_a[indices[0]], q_b[indices[0]])])

    latencies: list[float] = []
    for i in indices:
        start = perf_counter()
        vec_a = embedder.embed(q_a[i])
        vec_b = embedder.embed(q_b[i])
        sim = float(np.dot(vec_a, vec_b))
        if cross_model is not None and sim >= recall_threshold:
            cross_model.predict([(q_a[i], q_b[i])])
        latencies.append((perf_counter() - start) * 1000.0)
    return _mean_p95(latencies)


# ---------------------------------------------------------------------------
# Comparison + plotting
# ---------------------------------------------------------------------------
@dataclass
class ApproachResult:
    """One approach's sweep and headline operating points."""

    approach: str  # "bi-encoder" or "two-stage"
    label: str
    df: pd.DataFrame

    @property
    def best(self) -> pd.Series:
        return best_f1_row(self.df)

    @property
    def target(self) -> pd.Series | None:
        return precision_target_row(self.df, min_precision=0.95)


def _print_comparison(results: list[ApproachResult], ceiling: float) -> None:
    """Print the 95%-precision recall for every approach, side by side."""
    header = (
        f"{'approach':<12}{'model / stage':<48}"
        f"{'P95@thr':>9}{'recall@P95':>12}{'bestF1':>8}{'F1rec':>8}"
    )
    print("\n=== At 95% precision, what recall does each achieve? ===")
    print(f"(stage-1 recall ceiling = {ceiling:.3f} -- no approach can exceed it)")
    print(header)
    print("-" * len(header))
    for r in results:
        target = r.target
        if target is None:
            p95_thr, p95_rec = "n/a", "n/a"
        else:
            p95_thr = f"{float(target['threshold']):.3f}"
            p95_rec = f"{float(target['recall']):.3f}"
        best = r.best
        label = r.label if len(r.label) <= 47 else "..." + r.label[-44:]
        print(
            f"{r.approach:<12}{label:<48}{p95_thr:>9}{p95_rec:>12}"
            f"{float(best['f1']):>8.3f}{float(best['recall']):>8.3f}"
        )


def plot_pr_comparison(
    results: list[ApproachResult], path: Path, *, title_suffix: str = ""
) -> None:
    """Overlay bi-encoder-alone vs. two-stage precision-recall curves."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in results:
        style = "--" if r.approach == "bi-encoder" else "-"
        ax.plot(
            r.df["recall"], r.df["precision"],
            linestyle=style, marker="o", markersize=3, label=r.label,
        )
    ax.axhline(0.95, color="grey", linestyle=":", alpha=0.6, label="95% precision")
    ax.set_xlabel("Recall (duplicates caught, full sample)")
    ax.set_ylabel("Precision (cache hits that are correct)")
    title = "Bi-encoder alone vs. two-stage (bi-encoder + reranker)"
    ax.set_title(f"{title}\n{title_suffix}" if title_suffix else title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument(
        "--dataset", choices=["quora", "qqp", "paws"], default="qqp",
        help="Ground-truth source (default: qqp).",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to sample (train/validation/test; dataset-dependent).",
    )
    parser.add_argument(
        "--bi-encoder", default=None,
        help="Stage-1 bi-encoder (default: the app's EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--recall-threshold", type=float, default=0.70,
        help="Stage-1 cosine cut, tuned for recall (default: 0.70).",
    )
    parser.add_argument(
        "--cross-encoder", action="append", dest="cross_encoders", default=None,
        help="Stage-2 reranker(s); repeatable. Defaults to ms-marco + quora models.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument(
        "--latency-samples", type=int, default=200,
        help="Pairs timed for per-pair latency (default: 200).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory for CSV/plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bi_model = args.bi_encoder or default_model()
    cross_models = args.cross_encoders or DEFAULT_CROSS_ENCODERS
    # Namespace every output by dataset/split so runs never clobber each other.
    ds_slug = model_slug(f"{args.dataset}_{args.split}")

    print(
        f"Loading {args.n} pairs from '{args.dataset}' split '{args.split}' "
        f"(seed={args.seed})..."
    )
    q_a, q_b, labels = load_pairs(args.dataset, args.n, args.seed, args.split)
    n = int(labels.size)
    dupes = int(labels.sum())
    print(f"Sample: {n} pairs from {args.dataset}:{args.split} | duplicates {dupes} ({dupes / n:.1%})")

    # --- Stage 1: bi-encoder cosine + recall filter ----------------------
    print(f"\nStage 1 - bi-encoder '{bi_model}' (batch={args.batch_size})...")
    embedder = Embedder(bi_model)
    sims = compute_similarities(embedder, q_a, q_b, args.batch_size)
    survivors = survivors_mask(sims, args.recall_threshold)
    ceiling = recall_ceiling(labels, survivors)
    print(
        f"  survivors @ cos>={args.recall_threshold:.2f}: "
        f"{int(survivors.sum())}/{n} pairs "
        f"({int(survivors.sum()) / n:.1%})"
    )
    print(
        f"  RECALL CEILING (true duplicates surviving stage 1): {ceiling:.3f} "
        f"-- the two-stage pipeline can never exceed this."
    )

    # Single-stage bi-encoder baseline (same model, full sweep).
    df_bi = sweep_thresholds(sims, labels, THRESHOLDS)
    df_bi.to_csv(
        out_dir / f"threshold_sweep_{model_slug(bi_model)}_{ds_slug}.csv", index=False
    )
    results: list[ApproachResult] = [
        ApproachResult("bi-encoder", f"{bi_model} (stage 1 only)", df_bi)
    ]

    # --- Stage 2: each reranker ------------------------------------------
    survivor_idx = np.where(survivors)[0]
    survivor_pairs = [(q_a[i], q_b[i]) for i in survivor_idx]
    loaded_cross: dict[str, object] = {}
    for cross_name in cross_models:
        print(f"\nStage 2 - reranking {len(survivor_pairs)} survivors with '{cross_name}'...")
        probs, model = rerank(cross_name, survivor_pairs)
        loaded_cross[cross_name] = model
        df_two = two_stage_sweep(labels, survivors, probs)
        slug = model_slug(f"two_stage_{model_slug(bi_model)}__{cross_name}")
        csv_path = out_dir / f"threshold_sweep_{slug}_{ds_slug}.csv"
        df_two.to_csv(csv_path, index=False)
        print(f"  saved sweep -> {csv_path}")
        results.append(
            ApproachResult("two-stage", f"+ {cross_name}", df_two)
        )

    # --- Comparison + PR plot --------------------------------------------
    _print_comparison(results, ceiling)
    pr_path = out_dir / f"precision_recall_two_stage_{ds_slug}.png"
    plot_pr_comparison(
        results, pr_path, title_suffix=f"{args.dataset}:{args.split}"
    )
    print(f"\nSaved combined PR plot -> {pr_path}")

    # --- Latency: honest precision/latency tradeoff ----------------------
    print("\n=== Latency (ms per pair) ===")
    rng = np.random.default_rng(args.seed)
    k = min(args.latency_samples, n)
    idx = rng.choice(n, size=k, replace=False)

    s1_mean, s1_p95 = measure_latency(embedder, q_a, q_b, idx, args.recall_threshold)
    print(f"  stage 1 only                : mean {s1_mean:7.2f}  p95 {s1_p95:7.2f}")
    for cross_name, model in loaded_cross.items():
        m_mean, m_p95 = measure_latency(
            embedder, q_a, q_b, idx, args.recall_threshold, cross_model=model
        )
        print(f"  stage 1 + {cross_name[-28:]:<28}: mean {m_mean:7.2f}  p95 {m_p95:7.2f}")
    print(
        f"  (stage 2 runs only on the ~{int(survivors.sum()) / n:.0%} of pairs that "
        "clear stage 1, so the added cost is amortized.)"
    )


if __name__ == "__main__":
    main()
