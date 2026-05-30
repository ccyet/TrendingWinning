from __future__ import annotations

from inspect import getsource

import pandas as pd
import pytest

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


def test_legacy_backtest_applies_fee_and_slippage_to_actual_trade_prices() -> None:
    scanned = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]
            ),
            "stock_code": ["000001.SZ"] * 3,
            "open": [10.0, 10.4, 11.2],
            "high": [10.1, 10.8, 11.4],
            "low": [9.9, 10.2, 11.0],
            "close": [10.0, 10.6, 11.2],
            "volume": [1000.0, 1000.0, 1000.0],
            "amount": [10000.0, 10600.0, 11200.0],
            "breakout_trigger": [True, False, False],
            "trigger_price": [10.0, pd.NA, pd.NA],
        }
    )

    result = run_backtest(
        scanned,
        BacktestConfig(
            take_profit_pct=0.10,
            stop_loss_pct=0.05,
            max_holding_bars=2,
            fee_rate=0.001,
            slippage_bps=100.0,
        ),
    )

    trade = result.trades.iloc[0]
    expected_entry = 10.0 * 1.01
    expected_target = expected_entry * 1.10
    expected_exit = expected_target * 0.99
    expected_return_pct = ((expected_exit / expected_entry - 1.0) - 0.002) * 100.0

    assert trade["planned_entry_price"] == pytest.approx(10.0)
    assert trade["entry_price"] == pytest.approx(expected_entry)
    assert trade["target_price"] == pytest.approx(expected_target)
    assert trade["exit_price"] == pytest.approx(expected_exit)
    assert trade["return_pct"] == pytest.approx(expected_return_pct)
    assert result.stats["total_return"] == pytest.approx(expected_return_pct / 100.0)


def test_legacy_backtest_uses_single_full_position_across_symbols() -> None:
    rows: list[dict[str, object]] = []
    for symbol, base_price in (("000001.SZ", 10.0), ("000002.SZ", 20.0)):
        for index in range(4):
            price = base_price + index * 0.2
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": price,
                    "high": price + 0.3,
                    "low": price - 0.2,
                    "close": price,
                    "volume": 1000.0,
                    "amount": price * 1000.0,
                    "breakout_trigger": index == 0,
                    "trigger_price": base_price if index == 0 else pd.NA,
                }
            )
    scanned = pd.DataFrame(rows)

    result = run_backtest(
        scanned,
        BacktestConfig(take_profit_pct=0.20, stop_loss_pct=0.05, max_holding_bars=3),
    )

    assert result.trades["stock_code"].tolist() == ["000001.SZ"]
    assert result.stats["trade_count"] == 1.0


def test_legacy_backtest_reuses_vectorized_exit_scan_not_cursor_loop() -> None:
    source = getsource(_simulate_trade)

    assert "for cursor in range" not in source
