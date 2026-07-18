"""Offline evaluation of the semantic-cache similarity threshold.

Our ``SIMILARITY_THRESHOLD`` decides when two prompts are "the same question"
and a cached answer may be reused. This script replaces a hand-picked value
with a measured one: it scores our embedder against **human-labeled** question
pairs (Quora Question Pairs / GLUE QQP, where ``1 = duplicate``), sweeps the
threshold, and reports the precision/recall trade-off at each cut point.

It is fully offline - no LLM calls, no API, no running server. It imports the
project's real embedder (``app.embeddings.Embedder``, all-MiniLM-L6-v2) so the
measurement reflects exactly what the gateway uses in production.

Run::

    python eval/threshold_eval.py --n 5000 --seed 42

Outputs (under ``eval/results/``):
    * threshold_sweep.csv        - full per-threshold metrics
    * precision_recall_curve.png - PR curve across the sweep
    * precision_recall_vs_threshold.png - P/R/F1 as a function of threshold
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Make the project package importable when run as a script ------------
# `python eval/threshold_eval.py` puts eval/ on sys.path, not the repo root,
# so add the repo root explicitly before importing app.*.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.config import get_settings  # noqa: E402  (after sys.path fix)
from app.embeddings import Embedder  # noqa: E402  (after sys.path fix)

# Threshold grid: 0.60, 0.61, ... 0.99 inclusive. Built from ints to avoid
# floating-point drift in the labels.
THRESHOLDS: list[float] = [round(t / 100.0, 2) for t in range(60, 100)]

DEFAULT_OUT = _REPO_ROOT / "eval" / "results"


def default_model() -> str:
    """The app's configured embedding model (``EMBEDDING_MODEL`` / config default).

    Used as the ``--model`` default so the eval measures exactly what the gateway
    runs unless the caller overrides it to compare a different embedder.
    """
    return get_settings().embedding_model


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model name.

    Model names contain ``/`` (org/name) and sometimes ``:`` (revisions), which
    are not safe in filenames; collapse anything unsafe to ``_`` so per-model
    outputs sit side by side without clobbering each other.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return slug or "model"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _extract_pair(dataset: str, row: dict) -> tuple[str, str, int]:
    """Pull ``(text_a, text_b, label)`` from one row of any supported dataset.

    All three sources share binary label semantics (1 = duplicate / paraphrase,
    0 = not), which is exactly the ground truth a semantic cache needs.
    """
    if dataset == "quora":
        text_a, text_b = row["questions"]["text"]
        return text_a, text_b, int(row["is_duplicate"])
    if dataset == "qqp":
        return row["question1"], row["question2"], int(row["label"])
    if dataset == "paws":
        return row["sentence1"], row["sentence2"], int(row["label"])
    raise ValueError(f"Unknown dataset: {dataset!r}")


def load_pairs(
    dataset: str, n: int, seed: int, split: str = "train"
) -> tuple[list[str], list[str], np.ndarray]:
    """Load ``n`` labeled pairs from ``dataset``'s ``split``, seed-reproducible.

    Returns ``(text_a, text_b, labels)`` with ``labels`` a 0/1 int array
    (1 = human-labeled duplicate/paraphrase). Empty rows are skipped.

    Supported ``dataset`` / ``split``:

    * ``quora`` - Quora Question Pairs; single ``train`` split only.
    * ``qqp``   - GLUE QQP (the same underlying Quora data). ``train`` and
      ``validation`` are labeled; ``test`` labels are hidden (-1).
    * ``paws``  - PAWS ``labeled_final`` config; adversarial paraphrases with
      high lexical overlap. ``train`` / ``validation`` / ``test`` all labeled.
    """
    from datasets import load_dataset

    if dataset == "quora":
        if split != "train":
            raise ValueError(
                "The 'quora' dataset has only a 'train' split. Use --dataset "
                "qqp or paws to evaluate other splits."
            )
        # The `quora` builder is script-based, so it needs trust_remote_code
        # and `datasets < 4` (script loading was removed in datasets 4.x).
        try:
            ds = load_dataset("quora", trust_remote_code=True, split="train")
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to load the 'quora' dataset. It uses a loading script "
                "that requires `datasets<4` (see eval/requirements.txt, pinned "
                "to 3.2.0). Alternatively run with `--dataset qqp`, which uses "
                "the GLUE QQP config and works on current `datasets`.\n"
                f"Original error: {exc}"
            ) from exc
    elif dataset == "qqp":
        if split == "test":
            raise ValueError(
                "GLUE QQP 'test' labels are hidden (-1); use 'train' or "
                "'validation'."
            )
        ds = load_dataset("glue", "qqp", split=split)
    elif dataset == "paws":
        ds = load_dataset("paws", "labeled_final", split=split)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown dataset: {dataset!r}")

    ds = ds.shuffle(seed=seed)
    # Select a small buffer beyond n so skipped (empty) rows still leave >= n.
    pool = min(len(ds), n + max(64, n // 20))
    ds = ds.select(range(pool))

    q_a: list[str] = []
    q_b: list[str] = []
    labels: list[int] = []
    for row in ds:
        if len(labels) >= n:
            break
        text_a, text_b, label = _extract_pair(dataset, row)
        if not text_a or not text_b:
            continue
        q_a.append(text_a)
        q_b.append(text_b)
        labels.append(label)

    return q_a, q_b, np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Embedding + similarity
# ---------------------------------------------------------------------------
def _encode_all(
    embedder: Embedder, texts: list[str], batch_size: int
) -> np.ndarray:
    """Encode texts in batches; returns an ``(N, dim)`` float32 array."""
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        vectors.extend(embedder.embed_batch(chunk))
        print(f"    encoded {min(start + batch_size, total)}/{total}", end="\r")
    print()
    return np.asarray(vectors, dtype=np.float32)


def compute_similarities(
    embedder: Embedder,
    q_a: list[str],
    q_b: list[str],
    batch_size: int,
) -> np.ndarray:
    """Cosine similarity for each aligned pair.

    ``Embedder`` L2-normalizes its vectors, so the cosine similarity is the
    row-wise dot product. The result is clipped to ``[-1, 1]`` for safety.
    """
    print("  Encoding first questions...")
    vec_a = _encode_all(embedder, q_a, batch_size)
    print("  Encoding second questions...")
    vec_b = _encode_all(embedder, q_b, batch_size)
    sims = np.sum(vec_a * vec_b, axis=1)
    return np.clip(sims, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def sweep_thresholds(
    sims: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
    round_ndigits: int = 2,
) -> pd.DataFrame:
    """Compute a confusion matrix and derived metrics at each threshold.

    ``round_ndigits`` controls only how the threshold is *labeled* in the output
    (metrics always use the exact value). The 0.60-0.99 bi-encoder grid needs 2;
    cross-encoder thresholds cluster near 1.0 and need more to stay distinct.

    ``predicted duplicate := similarity >= threshold``. Columns:

    * precision, recall, f1              - standard classification metrics
    * false_hit_rate = FP / pred_pos     - share of cache hits that are wrong
                                           (== 1 - precision); this is the risk
    * duplicate_detection_rate = pred_pos / N
                                         - share of ALL pairs flagged as a
                                           duplicate, i.e. the real-world cache
                                           hit rate (distinct from recall,
                                           which is TP / actual duplicates)
    """
    is_pos = labels == 1
    is_neg = ~is_pos
    total = int(labels.size)
    actual_pos = int(is_pos.sum())

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = sims >= threshold
        tp = int(np.sum(predicted & is_pos))
        fp = int(np.sum(predicted & is_neg))
        tn = int(np.sum(~predicted & is_neg))
        fn = int(np.sum(~predicted & is_pos))

        predicted_pos = tp + fp
        precision = tp / predicted_pos if predicted_pos else 0.0
        recall = tp / actual_pos if actual_pos else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        false_hit_rate = fp / predicted_pos if predicted_pos else 0.0
        duplicate_detection_rate = predicted_pos / total if total else 0.0

        rows.append(
            {
                "threshold": round(threshold, round_ndigits),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_hit_rate": false_hit_rate,
                "duplicate_detection_rate": duplicate_detection_rate,
            }
        )
    return pd.DataFrame(rows)


def best_f1_row(df: pd.DataFrame) -> pd.Series:
    """Return the sweep row with the maximum F1."""
    return df.loc[df["f1"].idxmax()]


def precision_target_row(
    df: pd.DataFrame, min_precision: float = 0.95
) -> pd.Series | None:
    """Return the highest-recall row whose precision meets ``min_precision``.

    That operating point catches the most duplicates while keeping the wrong-
    hit rate at or below ``1 - min_precision``. ``None`` if unattainable.
    """
    eligible = df[df["precision"] >= min_precision]
    if eligible.empty:
        return None
    # Highest recall wins; break ties toward the lower (more permissive) cut.
    best = eligible.sort_values(
        ["recall", "threshold"], ascending=[False, True]
    ).iloc[0]
    return best


# ---------------------------------------------------------------------------
# Similarity distribution
# ---------------------------------------------------------------------------
def distribution_stats(
    sims: np.ndarray, labels: np.ndarray
) -> dict[str, dict[str, float]]:
    """Summary stats of the similarity score, split by ground-truth class."""
    out: dict[str, dict[str, float]] = {}
    for name, mask in (("duplicate", labels == 1), ("non_duplicate", labels == 0)):
        scores = sims[mask]
        if scores.size == 0:
            out[name] = {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
            continue
        out[name] = {
            "count": int(scores.size),
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p10": float(np.percentile(scores, 10)),
            "p90": float(np.percentile(scores, 90)),
        }
    return out


def class_separation_gap(stats: dict[str, dict[str, float]]) -> float:
    """How cleanly the two classes separate: duplicate median - non-dup p90.

    A positive gap means the typical duplicate scores higher than 90% of
    non-duplicates - the wider the gap, the easier it is to pick a threshold
    that catches duplicates without false hits. Can be negative when the
    distributions overlap badly (a poor embedder for this task).
    """
    return stats["duplicate"]["median"] - stats["non_duplicate"]["p90"]


def evaluate(
    model_name: str,
    q_a: list[str],
    q_b: list[str],
    labels: np.ndarray,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Embed with ``model_name`` and return ``(sweep_dataframe, dist_stats)``.

    The single reusable unit of work: build the embedder, score the fixed pair
    sample, sweep thresholds, and summarize the score distribution. Both the
    single-model CLI and the multi-model comparison call this so every embedder
    is measured identically on the same data.
    """
    embedder = Embedder(model_name)
    sims = compute_similarities(embedder, q_a, q_b, batch_size)
    df = sweep_thresholds(sims, labels, THRESHOLDS)
    stats = distribution_stats(sims, labels)
    return df, stats


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_precision_recall(
    df: pd.DataFrame, path: Path, *, title_suffix: str = ""
) -> None:
    """Save a precision-recall curve traced across the threshold sweep."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df["recall"], df["precision"], marker="o", markersize=3)
    ax.set_xlabel("Recall (duplicates caught)")
    ax.set_ylabel("Precision (cache hits that are correct)")
    title = "Precision-Recall across similarity thresholds"
    ax.set_title(f"{title}\n{title_suffix}" if title_suffix else title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metrics_vs_threshold(
    df: pd.DataFrame,
    path: Path,
    *,
    best_f1_threshold: float | None = None,
    precision_threshold: float | None = None,
    title_suffix: str = "",
) -> None:
    """Save precision/recall/F1 as a function of the threshold."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["threshold"], df["precision"], label="precision", marker=".")
    ax.plot(df["threshold"], df["recall"], label="recall", marker=".")
    ax.plot(df["threshold"], df["f1"], label="F1", marker=".")
    if best_f1_threshold is not None:
        ax.axvline(
            best_f1_threshold, color="green", linestyle="--", alpha=0.7,
            label=f"best F1 @ {best_f1_threshold:.2f}",
        )
    if precision_threshold is not None:
        ax.axvline(
            precision_threshold, color="red", linestyle=":", alpha=0.7,
            label=f">=95% precision @ {precision_threshold:.2f}",
        )
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Score")
    title = "Precision / Recall / F1 vs. threshold"
    ax.set_title(f"{title}\n{title_suffix}" if title_suffix else title)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_sweep_table(df: pd.DataFrame) -> None:
    """Pretty-print the sweep as an aligned table."""
    display = df.copy()
    for col in ("precision", "recall", "f1", "false_hit_rate", "duplicate_detection_rate"):
        display[col] = display[col].map(lambda v: f"{v:.3f}")
    print(display.to_string(index=False))


def _print_distribution(stats: dict[str, dict[str, float]]) -> None:
    header = f"{'class':<14}{'count':>8}{'mean':>9}{'median':>9}{'p10':>9}{'p90':>9}"
    print(header)
    print("-" * len(header))
    for name, s in stats.items():
        print(
            f"{name:<14}{s['count']:>8}{s['mean']:>9.3f}"
            f"{s['median']:>9.3f}{s['p10']:>9.3f}{s['p90']:>9.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Number of pairs to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument(
        "--dataset", choices=["quora", "qqp", "paws"], default="quora",
        help="Ground-truth source: Quora Question Pairs, GLUE QQP, or PAWS.",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to sample (train/validation/test; dataset-dependent).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Any sentence-transformers model name "
        "(default: the app's configured EMBEDDING_MODEL).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory for CSV/plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model or default_model()
    slug = model_slug(model_name)
    # Namespace every output by dataset/split so runs never clobber each other.
    ds_slug = model_slug(f"{args.dataset}_{args.split}")
    title_suffix = f"{args.dataset}:{args.split} - {model_name}"

    print(
        f"Loading {args.n} pairs from '{args.dataset}' split '{args.split}' "
        f"(seed={args.seed})..."
    )
    q_a, q_b, labels = load_pairs(args.dataset, args.n, args.seed, args.split)

    n = int(labels.size)
    dupes = int(labels.sum())
    non_dupes = n - dupes
    print(
        f"\nSample: {n} pairs from {args.dataset}:{args.split} | "
        f"duplicates: {dupes} ({dupes / n:.1%}) | "
        f"non-duplicates: {non_dupes} ({non_dupes / n:.1%})"
    )

    print(f"\nEmbedding with '{model_name}' (batch={args.batch_size})...")
    df, stats = evaluate(model_name, q_a, q_b, labels, args.batch_size)

    # --- Persist sweep ---------------------------------------------------
    csv_path = out_dir / f"threshold_sweep_{slug}_{ds_slug}.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Threshold sweep ===")
    _print_sweep_table(df)
    print(f"\nSaved full sweep -> {csv_path}")

    # --- Plots -----------------------------------------------------------
    best = best_f1_row(df)
    target = precision_target_row(df, min_precision=0.95)
    best_f1_threshold = float(best["threshold"])
    precision_threshold = float(target["threshold"]) if target is not None else None

    pr_path = out_dir / f"precision_recall_curve_{slug}_{ds_slug}.png"
    vt_path = out_dir / f"precision_recall_vs_threshold_{slug}_{ds_slug}.png"
    plot_precision_recall(df, pr_path, title_suffix=title_suffix)
    plot_metrics_vs_threshold(
        df, vt_path,
        best_f1_threshold=best_f1_threshold,
        precision_threshold=precision_threshold,
        title_suffix=title_suffix,
    )
    print(f"Saved plots  -> {pr_path}\n             -> {vt_path}")

    # --- Headline recommendations ---------------------------------------
    print("\n=== Recommendations ===")
    print(
        f"Best F1: threshold={best_f1_threshold:.2f}  "
        f"F1={best['f1']:.3f}  precision={best['precision']:.3f}  "
        f"recall={best['recall']:.3f}"
    )
    if target is not None:
        print(
            f">=95% precision: threshold={precision_threshold:.2f}  "
            f"precision={target['precision']:.3f}  recall={target['recall']:.3f}  "
            f"(false-hit rate {target['false_hit_rate']:.3f})"
        )
    else:
        print(">=95% precision: not achievable anywhere in the swept range.")
    print(
        "Class-separation gap (dup median - non-dup p90): "
        f"{class_separation_gap(stats):+.3f}"
    )

    print("\n=== Similarity distribution by class ===")
    _print_distribution(stats)


if __name__ == "__main__":
    main()
