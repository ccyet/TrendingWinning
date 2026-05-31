from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.models import BacktestConfig
from trending_winning.backtest.order_backtest import run_order_backtest, run_order_backtest_from_normalized


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": 1000.0,
                "amount": row["close"] * 1000.0,
            }
            for index, row in enumerate(
                [
                    {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
                    {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
                    {"open": 10.7, "high": 11.2, "low": 10.6, "close": 11.0},
                ]
            )
        ]
    )


def _orders(signal_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "order-runner",
                "event_id": "event-runner",
                "event_type": "trend_signal_bar",
                "strategy_name": "trend_signal_bar",
                "detector_name": "trend",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": signal_date,
                "signal_bar_index": 0,
                "side": "long",
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 2,
                "metadata": {},
            }
        ]
    )


def test_order_backtest_module_runs_independently_from_engine() -> None:
    bars = _bars()
    result = run_order_backtest(bars, _orders(bars.loc[0, "date"]), BacktestConfig(max_holding_bars=2))

    assert result.trades["order_id"].tolist() == ["order-runner"]
    assert result.order_decisions["status"].tolist() == ["accepted"]
    assert result.stats["market_bar_count"] == 3.0


def test_order_backtest_normalized_entry_does_not_renormalize(monkeypatch) -> None:
    from trending_winning.backtest import order_backtest as order_backtest_module
    from trending_winning.data.schema import normalize_bars

    normalized = normalize_bars(_bars())
    orders = _orders(normalized.loc[0, "date"])

    def fail_normalize(_: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("标准化订单回测入口不应重复 normalize。")

    monkeypatch.setattr(order_backtest_module, "normalize_bars", fail_normalize)
    result = run_order_backtest_from_normalized(normalized, orders, BacktestConfig(max_holding_bars=2))

    assert result.trades["order_id"].tolist() == ["order-runner"]


def test_engine_reexports_order_backtest_entrypoints_for_compatibility() -> None:
    from trending_winning.backtest import engine

    assert engine.run_order_backtest is run_order_backtest
    assert engine.run_order_backtest_from_normalized is run_order_backtest_from_normalized


def test_single_order_backtest_drawdown_uses_holding_period_price_path() -> None:
    bars = pd.DataFrame(
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
                "high": 10.1,
                "low": 10.0,
                "close": 10.1,
                "volume": 1000.0,
                "amount": 10100.0,
            },
            {
                "date": pd.Timestamp("2026-05-25 10:30:00"),
                "stock_code": "000001.SZ",
                "open": 10.1,
                "high": 10.5,
                "low": 8.08,
                "close": 10.1,
                "volume": 1000.0,
                "amount": 10100.0,
            },
            {
                "date": pd.Timestamp("2026-05-25 11:00:00"),
                "stock_code": "000001.SZ",
                "open": 10.1,
                "high": 10.5,
                "low": 10.0,
                "close": 10.1,
                "volume": 1000.0,
                "amount": 10100.0,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "path-drawdown",
                "event_id": "path-drawdown",
                "event_type": "trend_signal_bar",
                "strategy_name": "trend_signal_bar",
                "detector_name": "trend",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                "signal_bar_index": 0,
                "side": "long",
                "entry_price": 10.1,
                "stop_price": 7.0,
                "target_price": 12.0,
                "max_holding_bars": 2,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=2))

    assert result.trades["return_pct"].iloc[0] == pytest.approx(0.0)
    assert result.equity_curve["drawdown_net_value"].min() == pytest.approx(0.8)
    assert result.stats["max_drawdown"] == pytest.approx(-0.2)
    assert result.stats["max_drawdown_trough_at"] == "2026-05-25 10:30:00"
