from __future__ import annotations

from inspect import getsource

import pandas as pd
import pytest

from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS
from trending_winning.strategies.signal_bar import (
    SignalBarStopStrategy,
    SignalBarStopStrategyConfig,
    _signal_filter_decisions,
)


class FixedEventDetector:
    """固定事件识别器；用于单独验证策略层过滤，不牵涉 detector 算法。"""

    name = "fixed"

    def __init__(self, events: list[dict[str, object]]) -> None:
        self._events = events

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return pd.DataFrame(self._events, columns=DETECTOR_EVENT_COLUMNS)


class BrokenEventDetector:
    """故意破坏事件契约，用于验证策略层能给出明确报错。"""

    name = "broken"

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "event_id": "broken-1",
                    "detector_name": "broken",
                    "stock_code": "000001.SZ",
                    "timeframe": "30m",
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "bar_index": 1,
                    "event_type": "broken_setup",
                    "entry_price": 10.0,
                    "stop_price": 9.8,
                    "confidence": 1.0,
                    "metadata": {},
                }
            ]
        )


def _event(
    *,
    event_id: str,
    entry_price: float,
    stop_price: float,
    signal_price: float,
    direction: str = "long",
    bar_index: int = 1,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "detector_name": "fixed",
        "stock_code": "000001.SZ",
        "timeframe": "30m",
        "date": pd.Timestamp("2026-05-25 10:00:00"),
        "bar_index": bar_index,
        "event_type": "fixed_setup",
        "direction": direction,
        "signal_price": signal_price,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "confidence": 1.0,
        "metadata": {},
    }


def test_signal_bar_strategy_preserves_risk_constraints_for_execution_layer() -> None:
    strategy = SignalBarStopStrategy(
        FixedEventDetector(
            [
                _event(event_id="accepted", entry_price=10.0, stop_price=9.8, signal_price=9.9),
                _event(event_id="too_wide", entry_price=10.0, stop_price=9.0, signal_price=9.9),
                _event(event_id="too_far", entry_price=10.5, stop_price=10.3, signal_price=9.9),
            ]
        ),
        SignalBarStopStrategyConfig(
            name="risk_filtered",
            risk_reward=2.0,
            max_actual_risk_pct=0.03,
            max_chase_pct=0.03,
        ),
    )

    orders = strategy.generate_orders(pd.DataFrame())

    assert orders["event_id"].tolist() == ["accepted", "too_wide", "too_far"]
    assert orders["event_type"].tolist() == ["fixed_setup", "fixed_setup", "fixed_setup"]
    assert orders["max_actual_risk_pct"].tolist() == [0.03, 0.03, 0.03]
    assert orders["max_chase_pct"].tolist() == [0.03, 0.03, 0.03]
    assert orders.loc[0, "metadata"]["actual_risk_pct"] == 0.02
    assert orders.loc[0, "metadata"]["chase_pct"] == 0.010101010101
    assert orders.loc[1, "metadata"]["actual_risk_pct"] == 0.1
    assert orders.loc[2, "metadata"]["chase_pct"] == 0.060606060606


def test_signal_bar_strategy_rejects_zero_liquidity_signal_bars_before_order_generation() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]
            ),
            "stock_code": ["000001.SZ"] * 3,
            "open": [9.8, 10.0, 10.3],
            "high": [10.0, 10.2, 10.6],
            "low": [9.7, 9.9, 10.2],
            "close": [9.9, 10.1, 10.5],
            "volume": [1000.0, 0.0, 1200.0],
            "amount": [9900.0, 0.0, 12600.0],
        }
    )
    strategy = SignalBarStopStrategy(
        FixedEventDetector(
            [
                _event(event_id="zero-signal", entry_price=10.2, stop_price=9.8, signal_price=10.1, bar_index=1),
                _event(event_id="liquid-signal", entry_price=10.6, stop_price=10.1, signal_price=10.5, bar_index=2),
            ]
        ),
        SignalBarStopStrategyConfig(name="signal_liquidity_checked"),
    )

    orders = strategy.generate_orders(bars)
    decisions = strategy.last_filter_decisions.set_index("event_id")

    assert orders["event_id"].tolist() == ["liquid-signal"]
    assert decisions.index.tolist() == ["zero-signal", "liquid-signal"]
    assert decisions.loc["zero-signal", "status"] == "rejected"
    assert decisions.loc["zero-signal", "reason"] == "signal_bar_no_liquidity"
    assert decisions.loc["liquid-signal", "status"] == "accepted"
    assert decisions.loc["liquid-signal", "reason"] == ""


def test_signal_bar_strategy_reuses_normalized_bars_for_liquidity_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    from trending_winning.strategies import signal_bar as signal_bar_module

    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]
            ),
            "stock_code": ["000001.SZ"] * 3,
            "open": [9.8, 10.0, 10.3],
            "high": [10.0, 10.2, 10.6],
            "low": [9.7, 9.9, 10.2],
            "close": [9.9, 10.1, 10.5],
            "volume": [1000.0, 0.0, 1200.0],
            "amount": [9900.0, 0.0, 12600.0],
        }
    )
    strategy = SignalBarStopStrategy(
        FixedEventDetector(
            [
                _event(event_id="zero-signal", entry_price=10.2, stop_price=9.8, signal_price=10.1, bar_index=1),
                _event(event_id="liquid-signal", entry_price=10.6, stop_price=10.1, signal_price=10.5, bar_index=2),
            ]
        )
    )

    def fail_normalize(_: pd.DataFrame) -> pd.DataFrame:
        raise AssertionError("已标准化 K 线不应在信号 K 流动性检查里重复 normalize。")

    monkeypatch.setattr(signal_bar_module, "normalize_bars", fail_normalize)

    orders = strategy.generate_orders(bars)
    decisions = strategy.last_filter_decisions.set_index("event_id")

    assert orders["event_id"].tolist() == ["liquid-signal"]
    assert decisions.loc["zero-signal", "reason"] == "signal_bar_no_liquidity"


def test_signal_bar_strategy_records_non_tradable_detector_events_in_filter_log() -> None:
    watch_event = {
        **_event(event_id="middle-watch", entry_price=10.2, stop_price=9.8, signal_price=10.0, bar_index=1),
        "direction": "watch",
        "event_type": "no_trade_middle",
    }
    strategy = SignalBarStopStrategy(
        FixedEventDetector(
            [
                watch_event,
                _event(event_id="long-setup", entry_price=10.6, stop_price=10.1, signal_price=10.5, bar_index=2),
            ]
        ),
        SignalBarStopStrategyConfig(name="non_tradable_logged"),
    )

    orders = strategy.generate_orders(pd.DataFrame())
    decisions = strategy.last_filter_decisions.set_index("event_id")

    assert orders["event_id"].tolist() == ["long-setup"]
    assert decisions.index.tolist() == ["middle-watch", "long-setup"]
    assert decisions.loc["middle-watch", "status"] == "rejected"
    assert decisions.loc["middle-watch", "reason"] == "non_tradable_direction"
    assert decisions.loc["long-setup", "status"] == "accepted"


def test_signal_bar_strategy_filters_orders_by_side_mode_and_keeps_decision_log() -> None:
    strategy = SignalBarStopStrategy(
        FixedEventDetector(
            [
                _event(event_id="long-setup", entry_price=10.6, stop_price=10.1, signal_price=10.5, direction="long"),
                _event(event_id="short-setup", entry_price=9.8, stop_price=10.2, signal_price=10.0, direction="short"),
            ]
        ),
        SignalBarStopStrategyConfig(name="long_only_strategy", side_mode="long_only"),
    )

    orders = strategy.generate_orders(pd.DataFrame())
    decisions = strategy.last_filter_decisions.set_index("event_id")

    assert orders["event_id"].tolist() == ["long-setup"]
    assert orders["side"].tolist() == ["long"]
    assert decisions.loc["long-setup", "status"] == "accepted"
    assert decisions.loc["short-setup", "status"] == "rejected"
    assert decisions.loc["short-setup", "reason"] == "side_mode_filtered"


def test_signal_bar_strategy_rejects_unknown_side_mode() -> None:
    with pytest.raises(ValueError, match="side_mode 仅支持 both、long_only 或 short_only"):
        SignalBarStopStrategy(FixedEventDetector([]), SignalBarStopStrategyConfig(side_mode="bad-mode"))


def test_signal_bar_strategy_rejects_detector_events_missing_standard_columns() -> None:
    strategy = SignalBarStopStrategy(BrokenEventDetector())

    with pytest.raises(ValueError, match="detector 事件缺少字段.*direction.*signal_price"):
        strategy.generate_orders(pd.DataFrame())


def test_signal_bar_strategy_uses_vectorized_order_generation_not_dataframe_row_cursor() -> None:
    source = getsource(SignalBarStopStrategy.generate_orders)

    assert ".iterrows(" not in source


def test_signal_filter_decisions_uses_vectorized_columns_not_record_loop() -> None:
    source = getsource(_signal_filter_decisions)

    assert ".to_dict(" not in source
