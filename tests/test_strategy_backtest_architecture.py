from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.engine import BacktestConfig, run_single_strategy_backtest
from trending_winning.detectors.range import RangeDetector, RangeDetectorConfig
from trending_winning.detectors.trend import TrendDetector, TrendDetectorConfig
from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.signal_bar import SignalBarStopStrategy, SignalBarStopStrategyConfig


class FixedOrderStrategy:
    name = "fixed_order"

    def __init__(self, orders: list[dict[str, object]]) -> None:
        self._orders = orders

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return pd.DataFrame(self._orders, columns=ORDER_COLUMNS)


def _trend_bars() -> pd.DataFrame:
    rows = []
    close = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 10.9, 10.8, 11.1, 11.5, 11.9, 12.4, 12.9]
    for index, value in enumerate(close):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": value - (0.12 if index not in {8, 9} else 0.35),
                "high": value + (0.18 if index not in {8, 9} else 0.08),
                "low": value - (0.18 if index not in {8, 9} else 0.62),
                "close": value,
                "volume": 1000.0 + index * 25.0,
                "amount": value * (1000.0 + index * 25.0),
            }
        )
    return pd.DataFrame(rows)


def _range_bars() -> pd.DataFrame:
    rows = []
    close = [10.0, 10.3, 10.1, 10.4, 10.2, 10.5, 10.3, 10.45, 10.2, 10.35]
    for index, value in enumerate(close):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000002.SZ",
                "open": value - 0.05,
                "high": value + 0.18,
                "low": value - 0.18,
                "close": value,
                "volume": 1000.0,
                "amount": value * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _two_symbol_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    payload = {
        "000001.SZ": [
            ("2026-05-25 09:30:00", 10.0, 10.2, 9.8, 10.0),
            ("2026-05-25 10:00:00", 10.1, 10.2, 9.9, 10.0),
            ("2026-05-25 10:30:00", 10.0, 10.2, 9.0, 9.4),
        ],
        "000002.SZ": [
            ("2026-05-25 09:30:00", 20.0, 20.2, 19.8, 20.0),
            ("2026-05-25 10:00:00", 20.0, 21.5, 20.0, 21.0),
            ("2026-05-25 10:30:00", 21.0, 21.2, 20.8, 21.0),
        ],
    }
    for symbol, bars in payload.items():
        for date, open_price, high, low, close in bars:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "stock_code": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _risk_excursion_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]
            ),
            "stock_code": ["000001.SZ"] * 3,
            "open": [9.9, 10.0, 10.5],
            "high": [10.0, 10.6, 11.2],
            "low": [9.8, 9.8, 10.4],
            "close": [9.9, 10.5, 11.0],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [9900.0, 11550.0, 13200.0],
        }
    )


def _fixed_order(
    *,
    order_id: str,
    symbol: str,
    signal_date: str,
    signal_bar_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> dict[str, object]:
    return {
        "order_id": order_id,
        "strategy_name": "fixed_order",
        "detector_name": "fixed",
        "event_id": f"event:{order_id}",
        "stock_code": symbol,
        "timeframe": "30m",
        "signal_date": pd.Timestamp(signal_date),
        "signal_bar_index": signal_bar_index,
        "side": "long",
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "max_holding_bars": 2,
        "metadata": {},
    }


def test_detectors_emit_independent_standard_events() -> None:
    trend_events = TrendDetector(TrendDetectorConfig(lookback=5, min_trend_score=0.2)).detect(_trend_bars())
    range_events = RangeDetector(RangeDetectorConfig(lookback=6)).detect(_range_bars())

    assert not trend_events.empty
    assert set(trend_events["detector_name"]) == {"trend"}
    assert {"event_id", "event_type", "direction", "entry_price", "stop_price", "metadata"}.issubset(
        trend_events.columns
    )
    assert "range_pos" not in trend_events.columns

    assert set(range_events["detector_name"]) == {"range"}
    assert "no_trade_middle" in range_events["event_type"].unique()


def test_single_strategy_backtest_consumes_only_bound_detector_and_reports_richer_stats() -> None:
    strategy = SignalBarStopStrategy(
        detector=TrendDetector(TrendDetectorConfig(lookback=5, min_trend_score=0.2)),
        config=SignalBarStopStrategyConfig(name="trend_h2_only", risk_reward=1.0, max_holding_bars=4),
    )

    result = run_single_strategy_backtest(_trend_bars(), strategy, BacktestConfig(max_holding_bars=4))

    expected_trade_columns = {
        "order_id",
        "event_id",
        "event_type",
        "timeframe",
        "signal_date",
        "signal_bar_index",
        "side",
        "planned_entry_price",
        "stop_price",
        "target_price",
        "risk_per_share",
        "metadata",
    }
    assert expected_trade_columns.issubset(result.trades.columns)
    assert result.trades["order_id"].astype(str).str.startswith("trend_h2_only:trend:").all()
    assert result.trades["event_id"].astype(str).str.startswith("trend:").all()
    assert result.trades["event_type"].astype(str).str.startswith("bull_").all()
    assert result.trades["signal_bar_index"].ge(0).all()
    assert result.trades["risk_per_share"].gt(0).all()
    assert result.trades["strategy_name"].eq("trend_h2_only").all()
    assert result.trades["detector_name"].eq("trend").all()
    assert {"profit_factor", "expectancy", "avg_win", "avg_loss", "exposure_bars"}.issubset(result.stats)
    assert result.stats["trade_count"] >= 1


def test_single_strategy_backtest_orders_multi_symbol_trades_by_entry_time() -> None:
    strategy = FixedOrderStrategy(
        [
            _fixed_order(
                order_id="late-loser",
                symbol="000001.SZ",
                signal_date="2026-05-25 10:00:00",
                signal_bar_index=1,
                entry_price=10.0,
                stop_price=9.5,
                target_price=11.0,
            ),
            _fixed_order(
                order_id="early-winner",
                symbol="000002.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=20.5,
                stop_price=19.5,
                target_price=21.0,
            ),
        ]
    )

    result = run_single_strategy_backtest(_two_symbol_bars(), strategy, BacktestConfig(max_holding_bars=2))

    assert result.trades["order_id"].tolist() == ["early-winner", "late-loser"]
    assert result.trades["entry_date"].tolist() == sorted(result.trades["entry_date"].tolist())


def test_single_strategy_backtest_equity_statistics_start_from_initial_capital() -> None:
    strategy = FixedOrderStrategy(
        [
            _fixed_order(
                order_id="winner",
                symbol="000002.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=20.5,
                stop_price=19.5,
                target_price=21.0,
            ),
            _fixed_order(
                order_id="loser",
                symbol="000001.SZ",
                signal_date="2026-05-25 10:00:00",
                signal_bar_index=1,
                entry_price=10.0,
                stop_price=9.5,
                target_price=11.0,
            ),
        ]
    )

    result = run_single_strategy_backtest(_two_symbol_bars(), strategy, BacktestConfig(initial_equity=2.0))

    assert result.equity_curve.iloc[0]["trade_no"] == 0
    assert result.equity_curve.iloc[0]["net_value"] == 2.0
    assert "annualized_return" in result.stats
    assert "equity_sharpe" in result.stats
    expected_total_return = result.equity_curve.iloc[-1]["net_value"] / 2.0 - 1.0
    assert result.stats["total_return"] == pytest.approx(expected_total_return)


def test_single_strategy_backtest_records_trade_r_multiple_and_excursions() -> None:
    strategy = FixedOrderStrategy(
        [
            _fixed_order(
                order_id="risk-path",
                symbol="000001.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=10.0,
                stop_price=9.5,
                target_price=11.0,
            )
        ]
    )

    result = run_single_strategy_backtest(_risk_excursion_bars(), strategy, BacktestConfig(max_holding_bars=2))
    trade = result.trades.iloc[0]

    assert trade["exit_reason"] == "take_profit"
    assert trade["r_multiple"] == 2.0
    assert trade["mae_pct"] == -2.0
    assert trade["mfe_pct"] == 12.0
    assert trade["mae_r"] == -0.4
    assert trade["mfe_r"] == 2.4


def test_single_strategy_backtest_allows_next_entry_after_exit_on_signal_bar() -> None:
    strategy = FixedOrderStrategy(
        [
            _fixed_order(
                order_id="first-exits-on-entry-bar",
                symbol="000001.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=10.0,
                stop_price=9.5,
                target_price=10.5,
            ),
            _fixed_order(
                order_id="second-signals-on-exit-bar",
                symbol="000001.SZ",
                signal_date="2026-05-25 10:00:00",
                signal_bar_index=1,
                entry_price=10.5,
                stop_price=10.0,
                target_price=11.0,
            ),
        ]
    )

    result = run_single_strategy_backtest(_risk_excursion_bars(), strategy, BacktestConfig(max_holding_bars=2))

    assert result.trades["order_id"].tolist() == ["first-exits-on-entry-bar", "second-signals-on-exit-bar"]
    assert result.order_decisions["status"].tolist() == ["accepted", "accepted"]


def test_single_strategy_backtest_records_order_decisions_for_unfilled_and_overlapping_orders() -> None:
    strategy = FixedOrderStrategy(
        [
            _fixed_order(
                order_id="accepted",
                symbol="000001.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=10.0,
                stop_price=9.5,
                target_price=11.0,
            ),
            _fixed_order(
                order_id="overlap",
                symbol="000001.SZ",
                signal_date="2026-05-25 10:00:00",
                signal_bar_index=1,
                entry_price=10.2,
                stop_price=9.8,
                target_price=11.2,
            ),
            _fixed_order(
                order_id="not-filled",
                symbol="000002.SZ",
                signal_date="2026-05-25 09:30:00",
                signal_bar_index=0,
                entry_price=50.0,
                stop_price=49.0,
                target_price=52.0,
            ),
        ]
    )

    result = run_single_strategy_backtest(_two_symbol_bars(), strategy, BacktestConfig(max_holding_bars=2))
    decisions = result.order_decisions.set_index("order_id")

    assert decisions.loc["accepted", "status"] == "accepted"
    assert decisions.loc["accepted", "reason"] == ""
    assert decisions.loc["accepted", "entry_date"] == pd.Timestamp("2026-05-25 10:00:00")
    assert decisions.loc["overlap", "status"] == "rejected"
    assert decisions.loc["overlap", "reason"] == "already_open"
    assert decisions.loc["not-filled", "status"] == "rejected"
    assert decisions.loc["not-filled", "reason"] == "no_fill"
    assert result.stats["accepted_order_count"] == 1.0
    assert result.stats["rejected_order_count"] == 2.0
    assert result.stats["rejected_already_open_count"] == 1.0
    assert result.stats["rejected_no_fill_count"] == 1.0
