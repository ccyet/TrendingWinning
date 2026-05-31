from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.returns import (
    downside_deviation,
    max_return_streak,
    return_series_statistics,
)


def test_return_series_statistics_reports_distribution_and_path_metrics() -> None:
    returns = pd.Series([0.05, -0.02, 0.04, -0.01, -0.03, 0.02])

    stats = return_series_statistics(returns)

    return_p05 = returns.quantile(0.05)
    assert stats["avg_return"] == pytest.approx(returns.mean())
    assert stats["return_std"] == pytest.approx(returns.std(ddof=0))
    assert stats["downside_deviation"] == pytest.approx((returns.clip(upper=0.0).pow(2).mean()) ** 0.5)
    assert stats["best_trade"] == pytest.approx(0.05)
    assert stats["worst_trade"] == pytest.approx(-0.03)
    assert stats["return_p05"] == pytest.approx(return_p05)
    assert stats["return_p25"] == pytest.approx(returns.quantile(0.25))
    assert stats["return_p50"] == pytest.approx(returns.quantile(0.50))
    assert stats["return_p75"] == pytest.approx(returns.quantile(0.75))
    assert stats["return_p95"] == pytest.approx(returns.quantile(0.95))
    assert stats["cvar_95"] == pytest.approx(returns.loc[returns <= return_p05].mean())
    assert stats["max_consecutive_wins"] == 1.0
    assert stats["max_consecutive_losses"] == 2.0


def test_return_series_statistics_returns_stable_empty_values() -> None:
    stats = return_series_statistics(pd.Series(dtype=float))

    assert stats == {
        "avg_return": 0.0,
        "return_std": 0.0,
        "downside_deviation": 0.0,
        "max_consecutive_wins": 0.0,
        "max_consecutive_losses": 0.0,
        "best_trade": 0.0,
        "worst_trade": 0.0,
        "return_p05": 0.0,
        "return_p25": 0.0,
        "return_p50": 0.0,
        "return_p75": 0.0,
        "return_p95": 0.0,
        "cvar_95": 0.0,
    }


def test_max_return_streak_ignores_zero_return_breaks() -> None:
    returns = pd.Series([0.01, 0.02, 0.0, -0.01, -0.02, -0.03, 0.04])

    assert max_return_streak(returns, positive=True) == 2
    assert max_return_streak(returns, positive=False) == 3
    assert downside_deviation(pd.Series([0.03, 0.02])) == 0.0
