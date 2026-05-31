from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.terminal_false_breakout import (
    TerminalFalseBreakoutFilterConfig,
    TerminalFalseBreakoutFilterStrategy,
)


class FixedOrderStrategy:
    name = "fixed_signal_bar"

    def __init__(self, orders: list[dict[str, object]]) -> None:
        self._orders = orders

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return pd.DataFrame(self._orders, columns=ORDER_COLUMNS)


def _long_terminal_bars(*, future_reversal: bool = False) -> pd.DataFrame:
    closes = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.35, 13.55, 13.7, 13.82, 13.92]
    rows: list[dict[str, object]] = []
    for index, close in enumerate(closes):
        high = close + 0.18
        low = close - 0.20
        open_ = close - 0.08
        if index == len(closes) - 1:
            open_ = close - 0.02
            high = close + 1.10
            low = close - 0.18
        rows.append(_bar(index, open_, high, low, close))
    if future_reversal:
        rows.append(_bar(len(rows), 12.8, 13.0, 11.5, 11.8))
    return pd.DataFrame(rows)


def _short_terminal_bars() -> pd.DataFrame:
    closes = [20.0, 19.5, 19.0, 18.5, 18.0, 17.5, 17.0, 16.65, 16.45, 16.3, 16.18, 16.08]
    rows: list[dict[str, object]] = []
    for index, close in enumerate(closes):
        high = close + 0.20
        low = close - 0.18
        open_ = close + 0.08
        if index == len(closes) - 1:
            open_ = close + 0.02
            high = close + 0.18
            low = close - 1.10
        rows.append(_bar(index, open_, high, low, close))
    return pd.DataFrame(rows)


def _early_uptrend_bars() -> pd.DataFrame:
    closes = [10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5]
    return pd.DataFrame([_bar(index, close - 0.08, close + 0.18, close - 0.18, close) for index, close in enumerate(closes)])


def _bar(index: int, open_: float, high: float, low: float, close: float) -> dict[str, object]:
    return {
        "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
        "stock_code": "000001.SZ",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 1000.0 + index,
        "amount": close * (1000.0 + index),
    }


def _order(order_id: str, *, side: str, detector_name: str, signal_bar_index: int) -> dict[str, object]:
    if side == "long":
        signal_price = 14.0
        entry = 14.01
        stop = 13.0
        target = 16.0
        event_type = "bull_h2_setup"
    else:
        signal_price = 16.0
        entry = 15.99
        stop = 17.0
        target = 14.0
        event_type = "bear_l2_setup"
    return {
        "order_id": order_id,
        "strategy_name": "fixed_signal_bar",
        "detector_name": detector_name,
        "event_id": f"event:{order_id}",
        "event_type": event_type,
        "stock_code": "000001.SZ",
        "timeframe": "30m",
        "signal_date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * signal_bar_index),
        "signal_bar_index": signal_bar_index,
        "side": side,
        "signal_price": signal_price,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "max_holding_bars": 4,
        "max_actual_risk_pct": None,
        "max_chase_pct": None,
        "metadata": {},
    }


def _strict_config() -> TerminalFalseBreakoutFilterConfig:
    return TerminalFalseBreakoutFilterConfig(
        enabled=True,
        lookback=5,
        atr_period=3,
        min_regime_bars=4,
        extension_atr_multiple=0.6,
        edge_lookback=4,
        edge_pos=0.70,
        edge_min_count=2,
        weak_progress_atr=0.8,
        wick_ratio=0.30,
        min_score=3,
    )


def test_terminal_false_breakout_filter_rejects_late_trend_long_signal() -> None:
    bars = _long_terminal_bars()
    strategy = TerminalFalseBreakoutFilterStrategy(
        FixedOrderStrategy([_order("late-long", side="long", detector_name="trend", signal_bar_index=11)]),
        _strict_config(),
    )

    result = strategy.generate_order_plan(bars, timeframe="30m")

    assert result.orders.empty
    decision = result.filter_decisions.set_index("order_id").loc["late-long"]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "terminal_false_breakout_risk"
    assert decision["filter_name"] == "terminal_false_breakout_filter"


def test_terminal_false_breakout_filter_keeps_early_trend_signal() -> None:
    bars = _early_uptrend_bars()
    strategy = TerminalFalseBreakoutFilterStrategy(
        FixedOrderStrategy([_order("early-long", side="long", detector_name="trend", signal_bar_index=6)]),
        _strict_config(),
    )

    result = strategy.generate_order_plan(bars, timeframe="30m")

    assert result.orders["order_id"].tolist() == ["early-long"]
    assert result.filter_decisions.set_index("order_id").loc["early-long", "status"] == "accepted"


def test_terminal_false_breakout_filter_rejects_channel_upper_terminal_breakout() -> None:
    bars = _long_terminal_bars()
    strategy = TerminalFalseBreakoutFilterStrategy(
        FixedOrderStrategy([_order("late-channel", side="long", detector_name="channel", signal_bar_index=11)]),
        _strict_config(),
    )

    result = strategy.generate_order_plan(bars, timeframe="30m")

    assert result.orders.empty
    assert result.filter_decisions.set_index("order_id").loc["late-channel", "reason"] == "terminal_false_breakout_risk"


def test_terminal_false_breakout_filter_rejects_short_lower_terminal_breakdown() -> None:
    bars = _short_terminal_bars()
    strategy = TerminalFalseBreakoutFilterStrategy(
        FixedOrderStrategy([_order("late-short", side="short", detector_name="trend", signal_bar_index=11)]),
        _strict_config(),
    )

    result = strategy.generate_order_plan(bars, timeframe="30m")

    assert result.orders.empty
    assert result.filter_decisions.set_index("order_id").loc["late-short", "reason"] == "terminal_false_breakout_risk"


def test_terminal_false_breakout_filter_disabled_preserves_base_orders() -> None:
    bars = _long_terminal_bars()
    base_orders = pd.DataFrame(
        [_order("late-long", side="long", detector_name="trend", signal_bar_index=11)],
        columns=ORDER_COLUMNS,
    )
    strategy = TerminalFalseBreakoutFilterStrategy(
        FixedOrderStrategy(base_orders.to_dict("records")),
        TerminalFalseBreakoutFilterConfig(enabled=False),
    )

    result = strategy.generate_order_plan(bars, timeframe="30m")

    assert_frame_equal(result.orders.reset_index(drop=True), base_orders.reset_index(drop=True))


def test_terminal_false_breakout_filter_ignores_future_bars_after_signal() -> None:
    base = FixedOrderStrategy([_order("late-long", side="long", detector_name="trend", signal_bar_index=11)])
    strategy = TerminalFalseBreakoutFilterStrategy(base, _strict_config())

    without_future = strategy.generate_order_plan(_long_terminal_bars(), timeframe="30m").filter_decisions
    with_future = strategy.generate_order_plan(_long_terminal_bars(future_reversal=True), timeframe="30m").filter_decisions

    left = without_future.set_index("order_id").loc["late-long", ["status", "reason", "context_state"]]
    right = with_future.set_index("order_id").loc["late-long", ["status", "reason", "context_state"]]
    assert left.to_dict() == right.to_dict()
