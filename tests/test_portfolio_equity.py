from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.portfolio_equity import build_portfolio_equity_curve_from_normalized
from trending_winning.backtest.stats import compute_equity_statistics
from trending_winning.data.schema import normalize_bars


def test_portfolio_equity_curve_marks_short_cash_value_and_margin_exposure() -> None:
    bars = normalize_bars(
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.1,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000001.SZ",
                    "open": 9.0,
                    "high": 9.0,
                    "low": 9.0,
                    "close": 9.0,
                    "volume": 1000.0,
                    "amount": 9000.0,
                },
            ]
        )
    )
    trades = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:00:00"),
                "stock_code": "000001.SZ",
                "side": "short",
                "entry_price": 10.0,
                "raw_return_pct": 10.0,
                "capital_fraction": 0.5,
                "margin_fraction": 1.0,
                "portfolio_priority": 1,
            }
        ]
    )

    equity = build_portfolio_equity_curve_from_normalized(bars, trades, initial_equity=1.0)

    open_row = equity.loc[equity["open_positions"] == 1].iloc[0]
    assert open_row["cash"] == pytest.approx(1.5)
    assert open_row["position_value"] == pytest.approx(-0.5)
    assert open_row["net_value"] == pytest.approx(1.0)
    assert open_row["gross_exposure"] == pytest.approx(0.5)
    assert open_row["margin_exposure"] == pytest.approx(1.0)
    assert equity.iloc[-1]["cash"] == pytest.approx(1.05)
    assert equity.iloc[-1]["net_value"] == pytest.approx(1.05)


def test_portfolio_drawdown_uses_adverse_intrabar_asset_prices() -> None:
    bars = normalize_bars(
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.5,
                    "low": 8.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 12.0,
                    "high": 12.1,
                    "low": 11.8,
                    "close": 12.0,
                    "volume": 1000.0,
                    "amount": 12000.0,
                },
            ]
        )
    )
    trades = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:30:00"),
                "stock_code": "000001.SZ",
                "side": "long",
                "entry_price": 10.0,
                "raw_return_pct": 20.0,
                "capital_fraction": 1.0,
                "margin_fraction": 1.0,
                "portfolio_priority": 1,
            }
        ]
    )

    equity = build_portfolio_equity_curve_from_normalized(bars, trades, initial_equity=1.0)
    stats = compute_equity_statistics(equity, periods_per_year=3)

    assert equity["net_value"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.2])
    assert equity["drawdown_net_value"].tolist() == pytest.approx([1.0, 1.0, 0.8, 1.18])
    assert stats["total_return"] == pytest.approx(0.2)
    assert stats["max_drawdown"] == pytest.approx(-0.2)
    assert stats["max_drawdown_trough_at"] == "2026-05-25 10:00:00"


def test_portfolio_drawdown_uses_combined_open_position_price_path() -> None:
    bars = normalize_bars(
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000002.SZ",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 8.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000002.SZ",
                    "open": 10.0,
                    "high": 12.0,
                    "low": 9.8,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 12.0,
                    "high": 12.1,
                    "low": 11.9,
                    "close": 12.0,
                    "volume": 1000.0,
                    "amount": 12000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:30:00"),
                    "stock_code": "000002.SZ",
                    "open": 8.0,
                    "high": 8.1,
                    "low": 7.9,
                    "close": 8.0,
                    "volume": 1000.0,
                    "amount": 8000.0,
                },
            ]
        )
    )
    trades = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:30:00"),
                "stock_code": "000001.SZ",
                "side": "long",
                "entry_price": 10.0,
                "raw_return_pct": 20.0,
                "capital_fraction": 0.5,
                "margin_fraction": 0.5,
                "portfolio_priority": 1,
            },
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:30:00"),
                "stock_code": "000002.SZ",
                "side": "short",
                "entry_price": 10.0,
                "raw_return_pct": 20.0,
                "capital_fraction": 0.5,
                "margin_fraction": 0.5,
                "portfolio_priority": 2,
            },
        ]
    )

    equity = build_portfolio_equity_curve_from_normalized(bars, trades, initial_equity=1.0)
    stats = compute_equity_statistics(equity, periods_per_year=3)

    assert equity["net_value"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.2])
    assert equity["drawdown_net_value"].tolist() == pytest.approx([1.0, 1.0, 0.8, 1.19])
    assert stats["max_drawdown"] == pytest.approx(-0.2)
    assert stats["max_drawdown_trough_at"] == "2026-05-25 10:00:00"


def test_portfolio_drawdown_includes_exit_bar_asset_price_path_before_settlement() -> None:
    bars = normalize_bars(
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000001.SZ",
                    "open": 11.0,
                    "high": 11.2,
                    "low": 7.0,
                    "close": 11.0,
                    "volume": 1000.0,
                    "amount": 11000.0,
                },
            ]
        )
    )
    trades = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:00:00"),
                "stock_code": "000001.SZ",
                "side": "long",
                "entry_price": 10.0,
                "raw_return_pct": 10.0,
                "capital_fraction": 1.0,
                "margin_fraction": 1.0,
                "portfolio_priority": 1,
            }
        ]
    )

    equity = build_portfolio_equity_curve_from_normalized(bars, trades, initial_equity=1.0)
    stats = compute_equity_statistics(equity, periods_per_year=2)

    assert equity["net_value"].tolist() == pytest.approx([1.0, 1.0, 1.1])
    assert equity["drawdown_net_value"].tolist() == pytest.approx([1.0, 1.0, 0.7])
    assert stats["total_return"] == pytest.approx(0.1)
    assert stats["max_drawdown"] == pytest.approx(-0.3)
    assert stats["max_drawdown_trough_at"] == "2026-05-25 10:00:00"
    assert stats["max_drawdown_recovery_at"] == "2026-05-25 10:00:00"
    assert stats["current_drawdown"] == 0.0
