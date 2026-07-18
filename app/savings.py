"""Savings projection math — the single source of truth.

Shared by the live gateway (``app/metrics.py``) and the offline sensitivity
model (``eval/savings_model.py``) so the two can never drift apart. It lives in
``app/`` on purpose: ``app`` must not import from ``eval``, so the shared
function has to sit on the ``app`` side and be imported *into* the eval.

The model values the *cache* specifically: a hit serves an answer for $0 instead
of the average remote price, versus a no-cache baseline where every query pays
the remote model.
"""

from __future__ import annotations

MONTHS_PER_YEAR = 12


def project_savings(
    monthly_volume: float,
    hit_rate: float,
    avg_input_tokens: float,
    avg_output_tokens: float,
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> dict:
    """Project cache savings for one scenario. Pure: no I/O, no globals.

    ::

        per-query remote cost = (avg_input_tokens  * input_price_per_1m
                                 + avg_output_tokens * output_price_per_1m) / 1e6
        monthly savings       = monthly_volume * hit_rate * per-query remote cost
        annual savings        = monthly savings * 12

    Returns the per-query remote cost and the monthly/annual savings the cache
    produces versus a no-cache baseline.
    """
    per_query_remote_cost = (
        avg_input_tokens * input_price_per_1m
        + avg_output_tokens * output_price_per_1m
    ) / 1_000_000.0
    monthly_savings = monthly_volume * hit_rate * per_query_remote_cost
    return {
        "monthly_volume": monthly_volume,
        "hit_rate": hit_rate,
        "per_query_remote_cost_usd": per_query_remote_cost,
        "monthly_savings_usd": monthly_savings,
        "annual_savings_usd": monthly_savings * MONTHS_PER_YEAR,
    }
