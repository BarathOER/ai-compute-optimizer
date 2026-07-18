"""Unit tests for the multi-scenario savings model (eval/savings_model.py)."""

from __future__ import annotations

from eval.savings_model import (
    AVG_INPUT_TOKENS,
    AVG_OUTPUT_TOKENS,
    INPUT_PRICE_PER_1M,
    MEASURED_HIT_RATE,
    OUTPUT_PRICE_PER_1M,
    SCENARIO_HIT_RATES,
    SCENARIO_VOLUMES,
    project_savings,
    sensitivity_grid,
)


def test_single_source_of_truth() -> None:
    """app and eval share ONE project_savings function object (no divergence)."""
    from app.savings import project_savings as app_project_savings
    import app.metrics as metrics_module

    assert project_savings is app_project_savings
    assert metrics_module.project_savings is app_project_savings


def test_project_savings_math() -> None:
    """per-query cost and monthly/annual savings follow the documented formula."""
    result = project_savings(
        monthly_volume=1_000_000,
        hit_rate=0.5,
        avg_input_tokens=200,
        avg_output_tokens=400,
        input_price_per_1m=1.50,
        output_price_per_1m=9.00,
    )
    # (200*1.5 + 400*9) / 1e6 = (300 + 3600) / 1e6 = 0.0039
    assert result["per_query_remote_cost_usd"] == 0.0039
    # 1,000,000 * 0.5 * 0.0039 = 1950/month
    assert result["monthly_savings_usd"] == 1_950.0
    assert result["annual_savings_usd"] == 1_950.0 * 12


def test_savings_scale_linearly() -> None:
    """Doubling volume or hit rate doubles the savings (pure linear model)."""
    base = project_savings(100_000, 0.30, 220, 300, 1.50, 9.00)
    double_volume = project_savings(200_000, 0.30, 220, 300, 1.50, 9.00)
    double_hit = project_savings(100_000, 0.60, 220, 300, 1.50, 9.00)

    assert double_volume["annual_savings_usd"] == 2 * base["annual_savings_usd"]
    assert double_hit["annual_savings_usd"] == 2 * base["annual_savings_usd"]


def test_zero_hit_rate_saves_nothing() -> None:
    """A cache that never hits saves nothing, regardless of volume."""
    assert project_savings(1_000_000, 0.0, 220, 300, 1.50, 9.00)[
        "annual_savings_usd"
    ] == 0.0


def test_sensitivity_grid_shape_and_measured_column() -> None:
    """Grid has one row per volume and a cell per hit rate, incl. the measured one."""
    grid = sensitivity_grid(SCENARIO_VOLUMES, SCENARIO_HIT_RATES)
    assert len(grid) == len(SCENARIO_VOLUMES)
    assert MEASURED_HIT_RATE in SCENARIO_HIT_RATES

    for row, volume in zip(grid, SCENARIO_VOLUMES):
        assert row["monthly_volume"] == volume
        # Every scenario hit rate has a computed annual-savings cell.
        for hit_rate in SCENARIO_HIT_RATES:
            assert row[hit_rate] >= 0.0
        # Savings increase with hit rate within a row.
        ordered = [row[h] for h in sorted(SCENARIO_HIT_RATES)]
        assert ordered == sorted(ordered)


def test_grid_cell_matches_project_savings() -> None:
    """A grid cell equals the standalone project_savings for that scenario."""
    grid = sensitivity_grid([500_000], [MEASURED_HIT_RATE])
    expected = project_savings(
        500_000, MEASURED_HIT_RATE,
        AVG_INPUT_TOKENS, AVG_OUTPUT_TOKENS,
        INPUT_PRICE_PER_1M, OUTPUT_PRICE_PER_1M,
    )["annual_savings_usd"]
    assert grid[0][MEASURED_HIT_RATE] == expected
