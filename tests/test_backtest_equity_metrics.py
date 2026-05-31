from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.equity_metrics import (
    annualized_ratio,
    annualized_return,
    equity_return_statistics,
    infer_periods_per_year,
)


def test_equity_return_statistics_reports_annualized_return_and_ratios() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29"]),
            "net_value": [1.0, 1.02, 1.01, 1.05, 1.04],
        }
    )

    stats = equity_return_statistics(equity, equity["net_value"], max_drawdown=-0.02, periods_per_year=4)

    returns = equity["net_value"].pct_change().dropna()
    std = returns.std(ddof=0)
    downside = (returns.clip(upper=0.0).pow(2).mean()) ** 0.5
    assert stats["total_return"] == pytest.approx(0.04)
    assert stats["equity_return_std"] == pytest.approx(std)
    assert stats["equity_sharpe"] == pytest.approx(returns.mean() / std)
    assert stats["equity_sortino"] == pytest.approx(returns.mean() / downside)
    assert stats["annualized_return"] == pytest.approx(0.04)
    assert stats["annualized_volatility"] == pytest.approx(std * (4**0.5))
    assert stats["annualized_sharpe"] == pytest.approx(returns.mean() / std * (4**0.5))
    assert stats["annualized_sortino"] == pytest.approx(returns.mean() / downside * (4**0.5))
    assert stats["calmar_ratio"] == pytest.approx(0.04 / 0.02)


def test_infer_periods_per_year_uses_intraday_bar_spacing() -> None:
    equity = pd.DataFrame({"date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 09:35:00"])})

    assert infer_periods_per_year(equity) == pytest.approx(12096.0)


def test_annualized_helpers_return_zero_for_unusable_inputs() -> None:
    assert annualized_return(pd.Series([1.0]), 252.0, 0) == 0.0
    assert annualized_return(pd.Series([0.0, 1.0]), 252.0, 1) == 0.0
    assert annualized_ratio(pd.Series([0.01, 0.02]), 0.0, 252.0) == 0.0
    assert annualized_ratio(pd.Series(dtype=float), 1.0, 252.0) == 0.0


def test_equity_return_statistics_returns_stable_empty_values() -> None:
    stats = equity_return_statistics(pd.DataFrame(), pd.Series(dtype=float), max_drawdown=0.0)

    assert stats == {
        "total_return": 0.0,
        "equity_return_std": 0.0,
        "equity_sharpe": 0.0,
        "equity_sortino": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "annualized_sharpe": 0.0,
        "annualized_sortino": 0.0,
        "calmar_ratio": 0.0,
    }
