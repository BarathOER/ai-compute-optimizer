"""Multi-scenario savings model for the semantic cache.

The gateway's ``/metrics`` projection reports savings at a *single* assumed
volume. A single point is not a defensible business number, so this script
models a grid of scenarios and - crucially - labels every input as MEASURED or
ASSUMED so the output can never be mistaken for a guarantee.

The model is deliberately simple and pure (no I/O in :func:`project_savings`):

    per-query remote cost = (avg_input_tokens  * input_price_per_1m
                             + avg_output_tokens * output_price_per_1m) / 1e6
    monthly savings       = monthly_volume * hit_rate * per-query remote cost
    annual savings        = monthly savings * 12

It values the *cache* specifically: a hit serves an answer for $0 instead of the
average remote price, versus a no-cache baseline.

HONEST CAVEATS (also printed at runtime):
  * Local (Ollama) routes are modeled at $0. That is OPTIMISTIC - self-hosted
    inference still costs GPU/compute/electricity; only the marginal API price
    is zero. Savings here therefore reflect *avoided API spend*, not true margin.
  * The 44.4% hit rate is from a SYNTHETIC workload (QQP-validated two-stage eval
    + benchmark). Real hit rate depends entirely on how repetitive actual
    traffic is, and could be far lower or higher.
  * PAWS-style adversarial inputs (near-identical wording, inverted meaning) are
    a documented failure mode of the reranker; a workload heavy in those would
    both lower the hit rate and risk wrong hits.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# --- Make the project package importable when run as a script ------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Single source of truth for the projection math (app must not import eval).
from app.savings import project_savings  # noqa: E402

# ---------------------------------------------------------------------------
# Inputs, each tagged by provenance. MEASURED = observed/published; ASSUMED =
# a planning choice we cannot measure in advance.
# ---------------------------------------------------------------------------

# MEASURED - Google's published Gemini 3.5 Flash list price (USD per 1M tokens).
INPUT_PRICE_PER_1M = 1.50
OUTPUT_PRICE_PER_1M = 9.00
# MEASURED - average tokens per query from a benchmark.py workload (live run).
AVG_INPUT_TOKENS = 43.9
AVG_OUTPUT_TOKENS = 542.5
# MEASURED - hit rate from the QQP-validated two-stage eval + live benchmark.
MEASURED_HIT_RATE = 0.444

# ASSUMED - planning scenarios. Volume is a business input we cannot know; the
# non-measured hit rates bracket the measured one to show sensitivity.
SCENARIO_VOLUMES = [10_000, 100_000, 500_000, 1_000_000]
SCENARIO_HIT_RATES = [0.20, 0.30, MEASURED_HIT_RATE, 0.60]


def sensitivity_grid(
    volumes: list[int],
    hit_rates: list[float],
    *,
    avg_input_tokens: float = AVG_INPUT_TOKENS,
    avg_output_tokens: float = AVG_OUTPUT_TOKENS,
    input_price_per_1m: float = INPUT_PRICE_PER_1M,
    output_price_per_1m: float = OUTPUT_PRICE_PER_1M,
) -> list[dict]:
    """Return one row per volume; each row maps hit_rate -> annual savings USD."""
    grid: list[dict] = []
    for volume in volumes:
        row: dict = {"monthly_volume": volume}
        for hit_rate in hit_rates:
            row[hit_rate] = project_savings(
                volume, hit_rate, avg_input_tokens, avg_output_tokens,
                input_price_per_1m, output_price_per_1m,
            )["annual_savings_usd"]
        grid.append(row)
    return grid


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _fmt_volume(v: int) -> str:
    if v >= 1_000_000:
        return f"{v // 1_000_000}M"
    if v >= 1_000:
        return f"{v // 1_000}K"
    return str(v)


def _hit_rate_header(hit_rate: float) -> str:
    tag = " (measured)" if abs(hit_rate - MEASURED_HIT_RATE) < 1e-9 else ""
    return f"{hit_rate * 100:.1f}%{tag}"


def save_csv(grid: list[dict], hit_rates: list[float], path: Path) -> None:
    """Write the sensitivity grid (annual savings USD) to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["monthly_volume"] + [_hit_rate_header(h) for h in hit_rates])
        for row in grid:
            writer.writerow(
                [row["monthly_volume"]] + [f"{row[h]:.2f}" for h in hit_rates]
            )


def print_grid(grid: list[dict], hit_rates: list[float]) -> None:
    """Print the sensitivity grid as an aligned, readable table."""
    print("\n=== Annual savings (USD) by monthly volume x cache hit rate ===")
    col_headers = [_hit_rate_header(h) for h in hit_rates]
    header = f"{'volume':>10}" + "".join(f"{h:>18}" for h in col_headers)
    print(header)
    print("-" * len(header))
    for row in grid:
        cells = "".join(f"{'$' + format(row[h], ',.0f'):>18}" for h in hit_rates)
        print(f"{_fmt_volume(row['monthly_volume']):>10}{cells}")
    print("-" * len(header))


def print_assumptions() -> None:
    """State every input's provenance so the numbers are never read as fact."""
    print("\n=== Assumptions (every input tagged MEASURED or ASSUMED) ===")
    print(
        f"  MEASURED  token prices        : "
        f"${INPUT_PRICE_PER_1M:.2f}/1M in, ${OUTPUT_PRICE_PER_1M:.2f}/1M out "
        "(Gemini 3.5 Flash list price)"
    )
    print(
        f"  MEASURED  avg tokens/query    : {AVG_INPUT_TOKENS:.1f} in / "
        f"{AVG_OUTPUT_TOKENS:.1f} out (benchmark.py workload)"
    )
    print(
        f"  MEASURED  hit rate (mid col)  : {MEASURED_HIT_RATE * 100:.1f}% "
        "(QQP-validated two-stage eval + live benchmark)"
    )
    print("  ASSUMED   monthly volume      : the four scenario rows")
    print("  ASSUMED   other hit rates     : 20% / 30% / 60% bracket the measured one")
    print(
        "  ASSUMED   traffic distribution: real traffic resembles the "
        "benchmark/QQP mix"
    )


def print_caveats() -> None:
    """Print the honest limitations of the model."""
    print("\n=== Caveats ===")
    print(
        "  * Local (Ollama) routes are modeled at $0 -- OPTIMISTIC. Self-hosted\n"
        "    inference still costs GPU/compute; these figures are avoided API\n"
        "    spend, not true margin."
    )
    print(
        "  * The 44.4% hit rate is from a SYNTHETIC workload. Real hit rate\n"
        "    depends on how repetitive actual traffic is -- it could be far lower."
    )
    print(
        "  * PAWS-style adversarial inputs (near-identical wording, inverted\n"
        "    meaning) are a documented reranker failure mode; a workload heavy in\n"
        "    those would lower the hit rate and risk wrong hits."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "sensitivity.csv",
        help="Where to write the sensitivity CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    grid = sensitivity_grid(SCENARIO_VOLUMES, SCENARIO_HIT_RATES)

    print_grid(grid, SCENARIO_HIT_RATES)
    save_csv(grid, SCENARIO_HIT_RATES, args.out)
    print(f"\nSaved sensitivity grid -> {args.out}")

    print_assumptions()
    print_caveats()


if __name__ == "__main__":
    main()
