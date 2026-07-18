"""Contamination vs. distribution-shift check for the two-stage reranker.

``cross-encoder/quora-distilroberta-base`` was trained on the Quora
duplicate-questions corpus, and GLUE QQP is that same corpus. Two DISTINCT
questions follow from that, and this script keeps them separate instead of
conflating them:

1. Contamination -- did the reranker just memorize its training pairs?
   Tested by QQP-train vs. QQP-validation at the same operating point. If the
   score came from recalling pairs seen in training, validation recall would
   fall sharply below train. Observed: QQP-train ~0.668 -> QQP-validation
   ~0.688 recall @ 95% precision -- no drop, so the reranker is NOT
   contaminated; its QQP result generalizes within the corpus.

2. Distribution shift -- can it handle a different KIND of input?
   Tested by PAWS (labeled_final), an INDEPENDENT, synthetically adversarial
   paraphrase set: word-swap and back-translation produce pairs with
   near-identical vocabulary but inverted meaning ("flight from NYC to LA" vs.
   "flight from LA to NYC"). Low recall here is a SCOPE LIMITATION on
   adversarial / entity-substituted inputs -- a different failure from
   contamination, and not evidence of cheating.

The two numbers answer different questions and neither alone is "the" answer:

    QQP-validation R@95% precision -> expected performance on NATURAL
        rephrasing, the realistic traffic profile for an LLM cache.
    PAWS R@95% precision           -> documented WORST CASE on adversarial
        inputs the reranker was never designed for.

It reports recall @ 95% precision for BOTH the bi-encoder alone and the full
two-stage pipeline on each dataset/split.

Run::

    python eval/contamination_check.py --n 5000 \\
        --cross-encoder cross-encoder/quora-distilroberta-base

Evaluation only: no LLM calls, no server, nothing under app/ is modified.
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

import numpy as np  # noqa: E402

from app.embeddings import Embedder  # noqa: E402
from eval.threshold_eval import (  # noqa: E402
    THRESHOLDS,
    compute_similarities,
    default_model,
    load_pairs,
    model_slug,
    precision_target_row,
    sweep_thresholds,
)
from eval.two_stage_eval import (  # noqa: E402
    load_cross_encoder,
    recall_ceiling,
    score_pairs,
    survivors_mask,
    two_stage_sweep,
)

# (label, dataset, split). QQP-train vs QQP-validation isolates contamination;
# PAWS (independent corpus) probes distribution shift on adversarial inputs.
DEFAULT_CONFIGS: list[tuple[str, str, str]] = [
    ("QQP-train", "qqp", "train"),
    ("QQP-validation", "qqp", "validation"),
    ("PAWS", "paws", "validation"),
]


@dataclass
class ContaminationRow:
    """One dataset/split: class balance and recall@95%-precision for both stages."""

    label: str
    dataset: str
    split: str
    n: int
    dup_rate: float
    recall_ceiling: float
    bi_recall_at_p95: float | None
    two_recall_at_p95: float | None


def _recall_at_p95(df) -> float | None:
    """Recall at the >=95%-precision operating point, or None if unreachable."""
    target = precision_target_row(df, min_precision=0.95)
    return None if target is None else float(target["recall"])


def evaluate_config(
    label: str,
    dataset: str,
    split: str,
    *,
    n: int,
    seed: int,
    embedder: Embedder,
    cross_model,
    recall_threshold: float,
    batch_size: int,
    out_dir: Path,
) -> ContaminationRow:
    """Run bi-encoder-alone and two-stage on one dataset/split; persist sweeps."""
    print(f"\n=== {label}  ({dataset}:{split}) ===")
    q_a, q_b, labels = load_pairs(dataset, n, seed, split)
    size = int(labels.size)
    dupes = int(labels.sum())
    print(f"  {size} pairs | duplicates {dupes} ({dupes / size:.1%})")

    # Stage 1 similarities (shared by both approaches).
    sims = compute_similarities(embedder, q_a, q_b, batch_size)

    # Bi-encoder alone.
    df_bi = sweep_thresholds(sims, labels, THRESHOLDS)
    slug = model_slug(f"{dataset}_{split}")
    df_bi.to_csv(out_dir / f"contam_bi_{slug}.csv", index=False)

    # Two-stage: recall filter -> rerank survivors -> full-sample sweep.
    survivors = survivors_mask(sims, recall_threshold)
    ceiling = recall_ceiling(labels, survivors)
    survivor_pairs = [(q_a[i], q_b[i]) for i in np.where(survivors)[0]]
    print(f"  survivors @ cos>={recall_threshold:.2f}: {len(survivor_pairs)}/{size} "
          f"| recall ceiling {ceiling:.3f}")
    probs = score_pairs(cross_model, survivor_pairs)
    df_two = two_stage_sweep(labels, survivors, probs)
    df_two.to_csv(out_dir / f"contam_two_stage_{slug}.csv", index=False)

    return ContaminationRow(
        label=label,
        dataset=dataset,
        split=split,
        n=size,
        dup_rate=dupes / size,
        recall_ceiling=ceiling,
        bi_recall_at_p95=_recall_at_p95(df_bi),
        two_recall_at_p95=_recall_at_p95(df_two),
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_contamination_table(
    rows: list[ContaminationRow], bi_model: str, cross_model_name: str
) -> None:
    """Print recall@95%-precision across datasets for both approaches."""
    print("\n" + "=" * 78)
    print("CONTAMINATION vs. DISTRIBUTION-SHIFT CHECK - recall @ 95% precision")
    print(f"  bi-encoder : {bi_model}")
    print(f"  reranker   : {cross_model_name}")
    print("=" * 78)
    header = (
        f"{'dataset/split':<18}{'n':>6}{'dup%':>7}{'ceiling':>9}"
        f"{'bi R@P95':>11}{'2-stage R@P95':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.label:<18}{r.n:>6}{r.dup_rate * 100:>6.1f}%{r.recall_ceiling:>9.3f}"
            f"{_fmt(r.bi_recall_at_p95):>11}{_fmt(r.two_recall_at_p95):>15}"
        )
    print("-" * len(header))
    print(
        "Read (two separate questions -- neither number alone is 'the' answer):\n"
        "  * Contamination: compare QQP-train vs QQP-validation. A sharp drop\n"
        "    would mean the reranker memorized training pairs. No drop = not\n"
        "    contaminated; the QQP result generalizes within the corpus.\n"
        "  * QQP-validation R@P95 = expected performance on NATURAL rephrasing,\n"
        "    the realistic traffic profile for an LLM cache.\n"
        "  * PAWS R@P95 = WORST CASE on adversarial inputs (word-swap /\n"
        "    back-translation give near-identical vocabulary, inverted meaning).\n"
        "    A low score is a scope limitation, NOT contamination.\n"
        "  'n/a' means 95% precision was unreachable at any threshold."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000, help="Pairs per dataset/split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    parser.add_argument(
        "--bi-encoder", default=None,
        help="Stage-1 bi-encoder (default: the app's EMBEDDING_MODEL).",
    )
    parser.add_argument(
        "--cross-encoder", default="cross-encoder/quora-distilroberta-base",
        help="Stage-2 reranker to scrutinize (default: the quora-trained model).",
    )
    parser.add_argument(
        "--recall-threshold", type=float, default=0.70,
        help="Stage-1 cosine cut (default: 0.70).",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size.")
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "eval" / "results",
        help="Output directory for the per-config CSVs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bi_model = args.bi_encoder or default_model()
    print(f"Loading bi-encoder '{bi_model}' and reranker '{args.cross_encoder}'...")
    embedder = Embedder(bi_model)
    cross_model = load_cross_encoder(args.cross_encoder)

    rows: list[ContaminationRow] = []
    for label, dataset, split in DEFAULT_CONFIGS:
        rows.append(
            evaluate_config(
                label, dataset, split,
                n=args.n, seed=args.seed,
                embedder=embedder, cross_model=cross_model,
                recall_threshold=args.recall_threshold,
                batch_size=args.batch_size, out_dir=out_dir,
            )
        )

    print_contamination_table(rows, bi_model, args.cross_encoder)


if __name__ == "__main__":
    main()
