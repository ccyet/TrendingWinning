from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.equity_exposure import equity_exposure_statistics


def test_equity_exposure_statistics_reports_position_and_margin_usage() -> None:
    equity = pd.DataFrame(
        {
            "net_value": [1.0, 1.02, 1.01, 1.05, 1.04],
            "gross_exposure": [0.0, 0.5, 1.0, 0.25, 0.0],
            "margin_exposure": [0.0, 0.5, 1.5, 0.25, 0.0],
            "open_positions": [0, 1, 2, 1, 0],
        }
    )

    stats = equity_exposure_statistics(equity, equity["net_value"])

    assert stats["avg_gross_exposure"] == pytest.approx(0.35)
    assert stats["max_gross_exposure"] == pytest.approx(1.0)
    assert stats["avg_margin_exposure"] == pytest.approx(0.45)
    assert stats["max_margin_exposure"] == pytest.approx(1.5)
    assert stats["exposure_bar_ratio"] == pytest.approx(3 / 5)
    assert stats["avg_open_positions"] == pytest.approx(0.8)
    assert stats["max_open_positions"] == pytest.approx(2.0)


def test_equity_exposure_statistics_reports_cash_and_net_exposure_ratios() -> None:
    equity = pd.DataFrame(
        {
            "net_value": [1.0, 1.1, 1.2, 1.05],
            "cash": [1.0, 0.3, 1.55, 0.5],
            "position_value": [0.0, 0.8, -0.35, 0.55],
        }
    )

    stats = equity_exposure_statistics(equity, equity["net_value"])

    cash_ratio = equity["cash"] / equity["net_value"]
    net_exposure = equity["position_value"] / equity["net_value"]
    assert stats["avg_cash_ratio"] == pytest.approx(cash_ratio.mean())
    assert stats["min_cash_ratio"] == pytest.approx(cash_ratio.min())
    assert stats["max_cash_ratio"] == pytest.approx(cash_ratio.max())
    assert stats["avg_net_exposure"] == pytest.approx(net_exposure.mean())
    assert stats["min_net_exposure"] == pytest.approx(net_exposure.min())
    assert stats["max_net_exposure"] == pytest.approx(net_exposure.max())


def test_equity_exposure_statistics_zeroes_invalid_net_value_ratios() -> None:
    equity = pd.DataFrame({"net_value": [1.0, 0.0], "cash": [0.5, 1.0], "position_value": [0.5, 1.0]})

    stats = equity_exposure_statistics(equity, equity["net_value"])

    assert stats["avg_cash_ratio"] == pytest.approx(0.25)
    assert stats["max_cash_ratio"] == pytest.approx(0.5)
    assert stats["avg_net_exposure"] == pytest.approx(0.25)
    assert stats["max_net_exposure"] == pytest.approx(0.5)


def test_equity_exposure_statistics_returns_stable_empty_values() -> None:
    stats = equity_exposure_statistics(pd.DataFrame(), pd.Series(dtype=float))

    assert stats == {
        "avg_gross_exposure": 0.0,
        "max_gross_exposure": 0.0,
        "avg_margin_exposure": 0.0,
        "max_margin_exposure": 0.0,
        "exposure_bar_ratio": 0.0,
        "avg_open_positions": 0.0,
        "max_open_positions": 0.0,
        "avg_cash_ratio": 0.0,
        "min_cash_ratio": 0.0,
        "max_cash_ratio": 0.0,
        "avg_net_exposure": 0.0,
        "min_net_exposure": 0.0,
        "max_net_exposure": 0.0,
    }
