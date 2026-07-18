# Semantic-cache threshold evaluation

This is an **offline** harness that measures the right value for the gateway's
`SIMILARITY_THRESHOLD` against human-labeled data, instead of guessing it from a
handful of prompts.

## What it measures

The semantic cache reuses a cached answer when a new prompt is "the same
question" as a stored one — decided by cosine similarity ≥ `SIMILARITY_THRESHOLD`.
That is exactly a **duplicate-question classifier**, and the
[Quora Question Pairs](https://huggingface.co/datasets/quora) dataset provides
the ground truth: each row is two questions with a human label,
`1 = duplicate` (same meaning) / `0 = not duplicate`.

The script:

1. Loads a reproducible random sample (`--n`, default 5000; fixed `--seed`) and
   reports the class balance.
2. Embeds every question with the project's **own** embedder
   ([app/embeddings.py](../app/embeddings.py), `all-MiniLM-L6-v2`) — the same
   model the gateway uses — batching the encoding for speed.
3. Computes cosine similarity for each pair (vectors are L2-normalized, so this
   is a dot product).
4. Sweeps the threshold from **0.60 to 0.99** in 0.01 steps. At each cut it
   treats `similarity ≥ threshold` as "predicted duplicate" and computes:

   | Metric | Meaning |
   |--------|---------|
   | `tp` / `fp` / `tn` / `fn` | Confusion-matrix counts |
   | `precision` | Of predicted duplicates, share that really are (correct cache hits) |
   | `recall` | Of real duplicates, share detected |
   | `f1` | Harmonic mean of precision and recall |
   | `false_hit_rate` | `FP / predicted_positives` — share of cache hits that return a **wrong** answer (= `1 − precision`); the key risk |
   | `duplicate_detection_rate` | `predicted_positives / N` — share of **all** pairs flagged as duplicate, i.e. the real-world **cache hit rate** (distinct from recall) |

5. Prints the sweep table and writes the full results to
   `results/threshold_sweep.csv`.
6. Saves two plots to `results/`:
   `precision_recall_curve.png` and `precision_recall_vs_threshold.png`.
7. Prints the headline findings:
   - the threshold that **maximizes F1**,
   - the lowest-risk threshold reaching **≥ 95% precision** and the recall it
     buys (i.e. how many duplicates you still catch while keeping wrong hits
     ≤ 5%),
   - the **similarity distribution** for duplicates vs. non-duplicates
     (mean, median, p10, p90 per class) — the separation between these two
     distributions is what makes a good threshold exist at all.

### Reading the result

`false_hit_rate` is the business risk: a wrong cache hit means a user gets an
answer to a *different* question. Precision controls it directly
(`false_hit_rate = 1 − precision`). The usual choice is the **≥ 95% precision**
threshold — the most permissive cut that keeps wrong hits under 5% — rather than
the max-F1 point, which trades precision for recall. Compare either against the
current default of `0.85`.

## Running it

From the repository root:

```bash
pip install -r requirements.txt -r eval/requirements.txt
python eval/threshold_eval.py --n 5000 --seed 42
```

Options:

| Flag | Default | Purpose |
|------|---------|---------|
| `--n` | `5000` | Number of labeled pairs to sample |
| `--seed` | `42` | Random seed (reproducible sample) |
| `--dataset` | `quora` | `quora` or `qqp` (GLUE QQP) ground-truth source |
| `--model` | app's `EMBEDDING_MODEL` | Any sentence-transformers model to embed with |
| `--batch-size` | `256` | Embedding batch size |
| `--out` | `eval/results` | Output directory for the CSV and plots |

`--model` defaults to the gateway's configured `EMBEDDING_MODEL` (so by default
the eval measures exactly what runs in production) but accepts any
sentence-transformers name. Outputs are namespaced by **model, dataset, and
split** so runs never overwrite each other:
`threshold_sweep_{model}_{dataset}_{split}.csv`,
`precision_recall_curve_{model}_{dataset}_{split}.png`, and
`precision_recall_vs_threshold_{model}_{dataset}_{split}.png` (each component is
slugified for the filesystem).

> **Dataset note.** `--dataset quora` uses a script-based loader that requires
> `datasets < 4` (pinned to `3.2.0` in [requirements.txt](requirements.txt)). On
> newer `datasets`, use `--dataset qqp`, which loads the GLUE QQP config
> (same label semantics) and needs no remote code.

## Comparing embedders

[compare_models.py](compare_models.py) runs the sweep for several models against
**one** fixed sample (apples-to-apples) and prints a side-by-side table:

```bash
python eval/compare_models.py \
    sentence-transformers/all-MiniLM-L6-v2 \
    sentence-transformers/all-mpnet-base-v2 \
    BAAI/bge-small-en-v1.5 \
    --n 5000 --dataset qqp
```

For each model it reports the best-F1 threshold (with precision/recall), the
threshold needed for ≥ 95% precision (with the recall there), and the
**class-separation gap** = `duplicate median − non-duplicate p90`. The gap is a
threshold-free score of how cleanly the embedder separates duplicates from
non-duplicates — a larger gap means an easier, more robust threshold choice. It
also saves each model's sweep CSV
(`threshold_sweep_{model}_{dataset}_{split}.csv`) plus a combined
`precision_recall_comparison_{dataset}_{split}.png` overlaying one curve per model.

## Two-stage retrieval (bi-encoder → cross-encoder rerank)

Comparing three bi-encoders on QQP exposed an **architectural** wall: none reach
95% precision at usable recall (best was `bge-small` at ~11.6%) and every
class-separation gap was ≈ 0 (+0.010 … +0.026). Bi-encoders embed each question
*independently*, which erases the fine distinctions that flip meaning ("how do I
learn X" vs. "how do I teach X"). [two_stage_eval.py](two_stage_eval.py) tests
the standard fix — a **cross-encoder reranker** that reads both questions
jointly.

```bash
python eval/two_stage_eval.py --n 5000 --dataset qqp \
    --recall-threshold 0.70 \
    --cross-encoder cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --cross-encoder cross-encoder/quora-distilroberta-base
```

- **Stage 1** — the bi-encoder (`--bi-encoder`, default the app's
  `EMBEDDING_MODEL`) computes cosine similarity and keeps pairs above
  `--recall-threshold` (default `0.70`), a cut chosen for **recall, not
  precision**. The fraction of true duplicates that survive is the pipeline's
  **recall ceiling** — printed explicitly, because stage 2 can only discard
  candidates, never recover a dropped duplicate.
- **Stage 2** — each `--cross-encoder` reranks the survivors. The threshold is
  swept and scored against the **full sample** (stage-1 drops count as false
  negatives), so precision / recall / F1 / false-hit rate stay directly
  comparable to the single-stage runs above.

It defaults to two rerankers — `ms-marco-MiniLM-L-6-v2` (general relevance) and
`quora-distilroberta-base` (trained on this exact duplicate task) — and reports
both. Outputs:

- a per-pipeline sweep CSV
  (`threshold_sweep_two_stage_{model}_{dataset}_{split}.csv`),
- a comparison table answering *"at 95% precision, what recall does each
  approach reach?"* for the single-stage bi-encoder vs. each two-stage pipeline,
- a combined PR curve `precision_recall_two_stage_{dataset}_{split}.png`
  (bi-encoder alone vs. two-stage),
- **latency** (mean / p95 ms per pair) for stage 1 alone vs. stage 1 + stage 2,
  so the precision gain can be weighed honestly against its cost. Stage 2 is
  timed only on pairs that clear stage 1, matching production.

> The cross-encoders download from HuggingFace on first run (a few hundred MB
> each) and use `sentence-transformers`, already in the root `requirements.txt`.

## Contamination vs. distribution shift

`cross-encoder/quora-distilroberta-base` was *trained* on the Quora
duplicate-questions corpus — and **GLUE QQP is that same corpus**, just
repackaged by the GLUE benchmark. That raises **two distinct questions**, which
[contamination_check.py](contamination_check.py) deliberately keeps apart
instead of conflating. It measures recall @ 95% precision for both the
bi-encoder alone and the two-stage pipeline across three dataset/splits:

```bash
python eval/contamination_check.py --n 5000 \
    --cross-encoder cross-encoder/quora-distilroberta-base
```

**1. Contamination — did the reranker just memorize its training pairs?**
Tested by **QQP-train vs. QQP-validation**. If the score came from recalling
pairs seen in training, validation recall would drop sharply below train.

> **Result: no drop.** QQP-train `0.668` → QQP-validation `0.688` recall @ 95%
> precision. The reranker is **not contaminated** — its QQP result generalizes
> across the split, and QQP-validation is a fair measure of its ability on this
> corpus.

**2. Distribution shift — can it handle a different *kind* of input?**
Tested by **PAWS** (`labeled_final`), an **independent** corpus of
*synthetically adversarial* paraphrases. PAWS is built by word-swapping and
back-translation, which produce sentence pairs with near-identical vocabulary
but inverted meaning ("a flight from NYC to LA" vs. "a flight from LA to NYC").
Bi-encoder cosine — and to a large extent the reranker — keys on lexical
overlap, so these entity-swapped pairs are exactly the case it gets wrong.

> Failing PAWS is a **scope limitation**, not evidence of cheating. It is a
> *different* failure from contamination: the model was never trained for, and
> is not designed to catch, adversarial word-order/entity substitution.

### Neither number alone is "the" answer

The two results answer different questions, and you need both:

| Number | What it tells you |
|--------|-------------------|
| **QQP-validation** R@95% precision | Expected performance on **natural rephrasing** — the realistic traffic profile for an LLM semantic cache (users asking the same thing in different words). |
| **PAWS** R@95% precision | Documented **worst case** on adversarial / entity-substituted inputs the reranker was never designed for. |

For a typical cache serving natural user rephrasings, the QQP-validation number
is the one to plan around. **A deployment that must also stay correct on
adversarial or entity-swapped input cannot rely on this reranker alone** — it
would need a third tier: an **LLM judge** to adjudicate borderline reranker
scores, or a **domain-fine-tuned reranker** trained on hard, entity-substituted
negatives. Treat PAWS as the marker of that boundary, not as a verdict on the
QQP result.

### Choosing the split on any script

Every eval script takes `--dataset {quora,qqp,paws}` and `--split`
(`train`/`validation`/`test`, dataset-dependent) and prints exactly which
`dataset:split` the sample came from. Notes: `quora` has only a `train` split;
GLUE QQP `test` labels are hidden (use `train`/`validation`); PAWS labels all
three. The original single-stage and two-stage runs above drew from **`train`**
— which is why the check above separates the train/validation (contamination)
and PAWS (distribution-shift) axes.

## Savings sensitivity model

[savings_model.py](savings_model.py) replaces the gateway's single-point savings
projection with a **labeled, multi-scenario** model — because one assumed volume
and one hit rate is not a defensible business number.

```bash
python eval/savings_model.py
```

It prints an annual-savings grid (rows = monthly volume 10K/100K/500K/1M,
columns = hit rate 20% / 30% / **44.4% measured** / 60%), writes it to
`results/sensitivity.csv`, and prints an **Assumptions** section that tags every
input `MEASURED` or `ASSUMED`, plus the honest caveats (local routes modeled at
$0 is optimistic; the 44.4% hit rate is from a synthetic workload; PAWS-style
adversarial inputs are a documented failure mode). The core is a pure function,
`project_savings(...)`, so the math is unit-tested. The live `/metrics`
`projection` object carries the same provenance labels in its `basis` field, so
the API never presents the number as fact.

## Standalone

Every script here imports the embedder but makes **no** LLM calls, no API
requests, and needs no running server. None of them modify anything under
`app/`.
