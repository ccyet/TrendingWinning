from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.exposure import trade_exposure_statistics


def test_trade_exposure_statistics_reports_position_and_capital_efficiency() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [2.0, -1.5],
            "raw_return_pct": [10.0, -5.0],
            "capital_fraction": [0.2, 0.3],
            "margin_fraction": [0.2, 0.45],
            "holding_bars": [2, 3],
        }
    )

    stats = trade_exposure_statistics(trades, pd.Series([0.02, -0.015]))

    assert stats["exposure_bars"] == pytest.approx(5.0)
    assert stats["return_contribution"] == pytest.approx(0.005)
    assert stats["return_per_exposure_bar"] == pytest.approx(0.005 / 5.0)
    assert stats["capital_turnover"] == pytest.approx(0.5)
    assert stats["avg_capital_fraction"] == pytest.approx(0.25)
    assert stats["max_capital_fraction"] == pytest.approx(0.3)
    assert stats["margin_turnover"] == pytest.approx(0.65)
    assert stats["avg_margin_fraction"] == pytest.approx(0.325)
    assert stats["max_margin_fraction"] == pytest.approx(0.45)
    assert stats["capital_exposure_bars"] == pytest.approx(1.3)
    assert stats["margin_exposure_bars"] == pytest.approx(1.75)
    assert stats["avg_capital_exposure_per_trade"] == pytest.approx(0.65)
    assert stats["avg_margin_exposure_per_trade"] == pytest.approx(0.875)
    assert stats["return_per_capital_exposure_bar"] == pytest.approx(0.005 / 1.3)
    assert stats["return_per_margin_exposure_bar"] == pytest.approx(0.005 / 1.75)
    assert stats["capital_weighted_raw_return"] == pytest.approx(0.01)


def test_trade_exposure_statistics_keeps_unallocated_strategy_fields_zero() -> None:
    trades = pd.DataFrame({"return_pct": [5.0, -2.0, 1.0], "holding_bars": [3, 2, 1]})

    stats = trade_exposure_statistics(trades, pd.Series([0.05, -0.02, 0.01]))

    assert stats["exposure_bars"] == pytest.approx(6.0)
    assert stats["return_contribution"] == pytest.approx(0.04)
    assert stats["return_per_exposure_bar"] == pytest.approx(0.04 / 6.0)
    assert stats["capital_turnover"] == 0.0
    assert stats["margin_turnover"] == 0.0
    assert stats["return_per_capital_exposure_bar"] == 0.0
    assert stats["return_per_margin_exposure_bar"] == 0.0
    assert stats["capital_weighted_raw_return"] == 0.0


def test_trade_exposure_statistics_returns_stable_empty_values() -> None:
    stats = trade_exposure_statistics(pd.DataFrame(), pd.Series(dtype=float))

    assert set(stats) == {
        "exposure_bars",
        "return_contribution",
        "return_per_exposure_bar",
        "capital_turnover",
        "avg_capital_fraction",
        "max_capital_fraction",
        "margin_turnover",
        "avg_margin_fraction",
        "max_margin_fraction",
        "capital_exposure_bars",
        "margin_exposure_bars",
        "avg_capital_exposure_per_trade",
        "avg_margin_exposure_per_trade",
        "return_per_capital_exposure_bar",
        "return_per_margin_exposure_bar",
        "capital_weighted_raw_return",
    }
    assert all(value == 0.0 for value in stats.values())
