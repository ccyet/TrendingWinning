from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.risk_metrics import trade_risk_quality_statistics


def test_trade_risk_quality_statistics_reports_r_multiple_and_path_quality() -> None:
    trades = pd.DataFrame(
        {
            "r_multiple": [2.0, -1.0, 0.5, -0.5],
            "mae_pct": [-1.0, -2.0, -0.5, -1.5],
            "mfe_pct": [4.0, 1.0, 2.5, 0.8],
            "mae_r": [-0.5, -1.0, -0.25, -0.75],
            "mfe_r": [2.0, 0.5, 1.25, 0.4],
        }
    )

    stats = trade_risk_quality_statistics(trades)

    r_multiple = trades["r_multiple"]
    expected_sqn = (len(r_multiple) ** 0.5) * r_multiple.mean() / r_multiple.std(ddof=0)
    assert stats["avg_r_multiple"] == pytest.approx(0.25)
    assert stats["median_r_multiple"] == pytest.approx(0.0)
    assert stats["best_r_multiple"] == pytest.approx(2.0)
    assert stats["worst_r_multiple"] == pytest.approx(-1.0)
    assert stats["r_profit_factor"] == pytest.approx(2.5 / 1.5)
    assert stats["system_quality_number"] == pytest.approx(expected_sqn)
    assert stats["avg_mae_pct"] == pytest.approx(-1.25)
    assert stats["avg_mfe_pct"] == pytest.approx(2.075)
    assert stats["avg_mae_r"] == pytest.approx(-0.625)
    assert stats["avg_mfe_r"] == pytest.approx(1.0375)


def test_trade_risk_quality_statistics_returns_zero_for_missing_optional_columns() -> None:
    stats = trade_risk_quality_statistics(pd.DataFrame({"return_pct": [1.0, -1.0]}))

    assert stats == {
        "avg_r_multiple": 0.0,
        "median_r_multiple": 0.0,
        "best_r_multiple": 0.0,
        "worst_r_multiple": 0.0,
        "r_profit_factor": 0.0,
        "system_quality_number": 0.0,
        "avg_mae_pct": 0.0,
        "avg_mfe_pct": 0.0,
        "avg_mae_r": 0.0,
        "avg_mfe_r": 0.0,
    }


def test_trade_risk_quality_statistics_returns_stable_empty_values() -> None:
    stats = trade_risk_quality_statistics(pd.DataFrame())

    assert set(stats) == {
        "avg_r_multiple",
        "median_r_multiple",
        "best_r_multiple",
        "worst_r_multiple",
        "r_profit_factor",
        "system_quality_number",
        "avg_mae_pct",
        "avg_mfe_pct",
        "avg_mae_r",
        "avg_mfe_r",
    }
    assert all(value == 0.0 for value in stats.values())
