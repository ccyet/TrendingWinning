from __future__ import annotations

from inspect import getsource

import pandas as pd

from trending_winning.backtest.engine import BacktestConfig, _simulate_trade, run_backtest
from trending_winning.data.schema import empty_bars
from trending_winning.strategy import StrategyConfig, scan_bars


def _bars() -> pd.DataFrame:
    rows = []
    close = [10, 10.1, 10.3, 10.4, 10.7, 10.9, 11.0, 11.2, 11.3, 11.5, 11.6, 12.9, 13.4, 13.0]
    for index, value in enumerate(close):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": value - (0.1 if index != 11 else 0.9),
                "high": value + (0.2 if index != 11 else 0.5),
                "low": value - (0.2 if index != 11 else 1.0),
                "close": value,
                "volume": 1000.0 if index != 11 else 3000.0,
                "amount": value * (1000.0 if index != 11 else 3000.0),
            }
        )
    return pd.DataFrame(rows)


def test_scan_bars_returns_landmark_channel_and_trigger_columns() -> None:
    out = scan_bars(_bars(), StrategyConfig(channel_lookback=8, landmark_lookback=6))

    assert {"is_landmark", "channel_direction", "breakout_trigger"}.issubset(out.columns)
    assert out["breakout_trigger"].sum() >= 1


def test_run_backtest_opens_on_trigger_and_exits_on_target() -> None:
    scanned = scan_bars(_bars(), StrategyConfig(channel_lookback=8, landmark_lookback=6))

    result = run_backtest(scanned, BacktestConfig(take_profit_pct=0.03, stop_loss_pct=0.02, max_holding_bars=4))

    assert len(result.trades) == 1
    assert result.trades.loc[0, "exit_reason"] == "take_profit"
    assert result.stats["trade_count"] == 1
    assert result.stats["win_rate"] == 1.0
    assert result.stats["total_return"] > 0


def test_empty_local_bars_return_empty_scan_and_backtest_contract() -> None:
    scanned = scan_bars(empty_bars(), StrategyConfig())
    result = run_backtest(scanned, BacktestConfig())

    assert {"is_landmark", "channel_upper", "breakout_trigger"}.issubset(scanned.columns)
    assert scanned.empty
    assert result.trades.empty
    assert result.equity_curve.to_dict("records") == [{"trade_no": 0, "net_value": 1.0}]
    assert result.stats["trade_count"] == 0.0
    assert result.stats["total_return"] == 0.0
    assert result.stats["annualized_return"] == 0.0


def test_legacy_backtest_reuses_vectorized_exit_scan_not_cursor_loop() -> None:
    source = getsource(_simulate_trade)

    assert "for cursor in range" not in source
