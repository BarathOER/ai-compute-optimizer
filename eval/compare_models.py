"""Compare embedding models on the same semantic-cache threshold benchmark.

Runs the threshold sweep from :mod:`eval.threshold_eval` for several
sentence-transformers models against **one** fixed, human-labeled pair sample
(so the comparison is apples-to-apples), then prints a side-by-side table and
saves a combined precision-recall plot with one curve per model.

For each model it reports:
    * the best-F1 threshold and its precision/recall,
    * the threshold needed for >= 95% precision and the recall there,
    * the class-separation gap (duplicate median - non-duplicate p90), a
      threshold-free measure of how well the embedder separates the classes.

Run::

    python eval/compare_models.py \\
        sentence-transformers/all-MiniLM-L6-v2 \\
        sentence-transformers/all-mpnet-base-v2 \\
        BAAI/bge-small-en-v1.5 \\
        --n 5000 --dataset qqp

This is evaluation only: no LLM calls, no server, and nothing under app/ is
modified.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# --- Make the project package importable when run as a script ------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from eval.threshold_eval import (  # noqa: E402  (after sys.path fix)
    DEFAULT_OUT,
    best_f1_row,
    class_separation_gap,
    evaluate,
    load_pairs,
    model_slug,
    precision_target_row,
)


@dataclass
class ModelComparison:
    """One model's headline results on the shared benchmark."""

    model: str
    df: pd.DataFrame
    best_f1_threshold: float
    best_f1: float
    best_precision: float
    best_recall: float
    p95_threshold: float | None
    p95_recall: float | None
    separation_gap: float


def compare(
    models: list[str],
    q_a: list[str],
    q_b: list[str],
    labels,
    batch_size: int,
    out_dir: Path,
    ds_slug: str = "",
) -> list[ModelComparison]:
    """Evaluate every model on the shared sample; persist each sweep CSV.

    ``ds_slug`` (``dataset_split``) namespaces each CSV so runs on different
    datasets/splits do not clobber each other.
    """
    suffix = f"_{ds_slug}" if ds_slug else ""
    results: list[ModelComparison] = []
    for model in models:
        print(f"\n=== Evaluating: {model} ===")
        df, stats = evaluate(model, q_a, q_b, labels, batch_size)

        csv_path = out_dir / f"threshold_sweep_{model_slug(model)}{suffix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  saved sweep -> {csv_path}")

        best = best_f1_row(df)
        target = precision_target_row(df, min_precision=0.95)
        results.append(
            ModelComparison(
                model=model,
                df=df,
                best_f1_threshold=float(best["threshold"]),
                best_f1=float(best["f1"]),
                best_precision=float(best["precision"]),
                best_recall=float(best["recall"]),
                p95_threshold=None if target is None else float(target["threshold"]),
                p95_recall=None if target is None else float(target["recall"]),
                separation_gap=class_separation_gap(stats),
            )
        )
    return results


def _print_comparison(results: list[ModelComparison]) -> None:
    """Print an aligned side-by-side comparison table."""
    header = (
        f"{'model':<44}{'F1@thr':>8}{'F1':>7}{'prec':>7}{'rec':>7}"
        f"{'P95@thr':>9}{'P95 rec':>9}{'sep_gap':>9}"
    )
    print("\n=== Model comparison ===")
    print(header)
    print("-" * len(header))
    for r in results:
        p95_thr = "n/a" if r.p95_threshold is None else f"{r.p95_threshold:.2f}"
        p95_rec = "n/a" if r.p95_recall is None else f"{r.p95_recall:.3f}"
        # Keep long org/name identifiers readable without breaking alignment.
        name = r.model if len(r.model) <= 43 else "..." + r.model[-40:]
        print(
            f"{name:<44}{r.best_f1_threshold:>8.2f}{r.best_f1:>7.3f}"
            f"{r.best_precision:>7.3f}{r.best_recall:>7.3f}"
            f"{p95_thr:>9}{p95_rec:>9}{r.separation_gap:>+9.3f}"
        )


def plot_pr_comparison(
    results: list[ModelComparison], path: Path, *, title_suffix: str = ""
) -> None:
    """Save a precision-recall plot overlaying one curve per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in results:
        ax.plot(
            r.df["recall"], r.df["precision"], marker="o", markersize=3, label=r.model
        )
    ax.axhline(0.95, color="grey", linestyle=":", alpha=0.6, label="95% precision")
    ax.set_xlabel("Recall (duplicates caught)")
    ax.set_ylabel("Precision (cache hits that are correct)")
    title = "Precision-Recall by embedding model"
    ax.set_title(f"{title}\n{title_suffix}" if title_suffix else title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", nargs="+", help="sentence-transformers model names to compare."
    )
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
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="Output directory for CSV/plots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {args.n} pairs from '{args.dataset}' split '{args.split}' "
        f"(seed={args.seed})..."
    )
    q_a, q_b, labels = load_pairs(args.dataset, args.n, args.seed, args.split)
    n = int(labels.size)
    dupes = int(labels.sum())
    print(
        f"Shared sample: {n} pairs from {args.dataset}:{args.split} | "
        f"duplicates {dupes} ({dupes / n:.1%}) | comparing {len(args.models)} model(s)"
    )

    # Namespace every output by dataset/split so runs never clobber each other.
    ds_slug = model_slug(f"{args.dataset}_{args.split}")
    results = compare(
        args.models, q_a, q_b, labels, args.batch_size, out_dir, ds_slug
    )

    _print_comparison(results)

    combined = out_dir / f"precision_recall_comparison_{ds_slug}.png"
    plot_pr_comparison(results, combined, title_suffix=f"{args.dataset}:{args.split}")
    print(f"\nSaved combined PR plot -> {combined}")


if __name__ == "__main__":
    main()
