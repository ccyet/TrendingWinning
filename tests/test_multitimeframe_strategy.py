from __future__ import annotations

from inspect import getsource

import pandas as pd

from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.diagnostics import STRATEGY_FILTER_DECISION_COLUMNS
from trending_winning.strategies.multitimeframe import (
    HigherTimeframeAlignmentStrategy,
    TimeframeAlignmentConfig,
    _filter_decision_frame,
)


class FixedOrderStrategy:
    name = "fixed_signal_bar"

    def __init__(self, orders: list[dict[str, object]]) -> None:
        self._orders = orders

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return pd.DataFrame(self._orders, columns=ORDER_COLUMNS)


class FixedOrderStrategyWithBaseFilter(FixedOrderStrategy):
    """带基础过滤日志的固定策略；用于验证包装策略不会吞掉内层拒绝原因。"""

    def __init__(self, orders: list[dict[str, object]], filter_decisions: list[dict[str, object]]) -> None:
        super().__init__(orders)
        self.last_filter_decisions = pd.DataFrame(filter_decisions, columns=STRATEGY_FILTER_DECISION_COLUMNS)


def _order(order_id: str, *, symbol: str, side: str, signal_date: str) -> dict[str, object]:
    entry = 10.5 if side == "long" else 9.5
    stop = 9.8 if side == "long" else 10.2
    target = 11.9 if side == "long" else 8.1
    return {
        "order_id": order_id,
        "strategy_name": "fixed_signal_bar",
        "detector_name": "fixed",
        "event_id": f"event:{order_id}",
        "stock_code": symbol,
        "timeframe": "15m",
        "signal_date": pd.Timestamp(signal_date),
        "signal_bar_index": 1,
        "side": side,
        "signal_price": 10.0,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "max_holding_bars": 4,
        "max_actual_risk_pct": None,
        "max_chase_pct": None,
        "metadata": {},
    }


def _filter_decision(order_id: str, *, status: str, reason: str) -> dict[str, object]:
    return {
        "order_id": order_id,
        "event_id": f"event:{order_id}",
        "strategy_name": "fixed_signal_bar",
        "base_strategy_name": "fixed_signal_bar",
        "detector_name": "fixed",
        "stock_code": "000001.SZ",
        "timeframe": "15m",
        "signal_date": pd.Timestamp("2026-05-25 10:00:00"),
        "signal_bar_index": 1,
        "side": "long",
        "status": status,
        "reason": reason,
        "filter_name": "signal_bar_adapter",
        "context_timeframe": "",
        "context_date": pd.NaT,
        "context_state": "",
    }


def test_higher_timeframe_alignment_filters_orders_without_touching_detector_contract() -> None:
    base = FixedOrderStrategy(
        [
            _order("long-match", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:00:00"),
            _order("short-mismatch", symbol="000001.SZ", side="short", signal_date="2026-05-25 10:30:00"),
            _order("short-match", symbol="000002.SZ", side="short", signal_date="2026-05-25 10:30:00"),
            _order("future-context-only", symbol="000003.SZ", side="long", signal_date="2026-05-25 10:00:00"),
        ]
    )
    higher_context = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 09:30:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "direction": ["long", "bear", "long"],
        }
    )
    strategy = HigherTimeframeAlignmentStrategy(
        base,
        higher_context,
        TimeframeAlignmentConfig(name="mtf_aligned", context_timeframe="60m", context_column="direction"),
    )

    orders = strategy.generate_orders(pd.DataFrame(), timeframe="15m")

    assert orders["order_id"].tolist() == ["long-match", "short-match"]
    assert orders["strategy_name"].tolist() == ["mtf_aligned", "mtf_aligned"]
    assert orders["metadata"].iloc[0]["base_strategy_name"] == "fixed_signal_bar"
    assert orders["metadata"].iloc[0]["higher_timeframe"] == "60m"
    assert orders["metadata"].iloc[0]["higher_state"] == "long"
    assert orders["metadata"].iloc[1]["higher_state"] == "bear"


def test_higher_timeframe_alignment_can_reject_stale_context() -> None:
    base = FixedOrderStrategy(
        [
            _order("fresh", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:00:00"),
            _order("stale", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:30:00"),
        ]
    )
    higher_context = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-05-25 09:30:00")],
            "stock_code": ["000001.SZ"],
            "direction": ["long"],
        }
    )
    strategy = HigherTimeframeAlignmentStrategy(
        base,
        higher_context,
        TimeframeAlignmentConfig(context_timeframe="60m", max_context_age=pd.Timedelta(minutes=30)),
    )

    orders = strategy.generate_orders(pd.DataFrame(), timeframe="15m")

    assert orders["order_id"].tolist() == ["fresh"]


def test_higher_timeframe_alignment_records_filter_decisions_for_all_base_orders() -> None:
    base = FixedOrderStrategy(
        [
            _order("accepted", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:00:00"),
            _order("mismatch", symbol="000001.SZ", side="short", signal_date="2026-05-25 10:00:00"),
            _order("stale", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:20:00"),
            _order("no-context", symbol="000002.SZ", side="long", signal_date="2026-05-25 10:00:00"),
        ]
    )
    higher_context = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:45:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000002.SZ"],
            "direction": ["long", "long"],
        }
    )
    strategy = HigherTimeframeAlignmentStrategy(
        base,
        higher_context,
        TimeframeAlignmentConfig(
            name="mtf_aligned",
            context_timeframe="60m",
            context_column="direction",
            max_context_age=pd.Timedelta(minutes=30),
        ),
    )

    orders = strategy.generate_orders(pd.DataFrame(), timeframe="15m")
    decisions = strategy.last_filter_decisions.set_index("order_id")

    assert orders["order_id"].tolist() == ["accepted"]
    assert decisions.index.tolist() == ["accepted", "mismatch", "stale", "no-context"]
    assert decisions.loc["accepted", "status"] == "accepted"
    assert decisions.loc["accepted", "reason"] == ""
    assert decisions.loc["accepted", "context_timeframe"] == "60m"
    assert decisions.loc["accepted", "context_state"] == "long"
    assert decisions.loc["mismatch", "status"] == "rejected"
    assert decisions.loc["mismatch", "reason"] == "higher_timeframe_mismatch"
    assert decisions.loc["stale", "reason"] == "higher_timeframe_stale"
    assert decisions.loc["no-context", "reason"] == "higher_timeframe_no_context"


def test_higher_timeframe_alignment_preserves_base_strategy_filter_decisions() -> None:
    base = FixedOrderStrategyWithBaseFilter(
        [_order("accepted", symbol="000001.SZ", side="long", signal_date="2026-05-25 10:00:00")],
        [
            _filter_decision("base-rejected", status="rejected", reason="signal_bar_no_liquidity"),
            _filter_decision("accepted", status="accepted", reason=""),
        ],
    )
    higher_context = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-05-25 09:30:00")],
            "stock_code": ["000001.SZ"],
            "direction": ["long"],
        }
    )
    strategy = HigherTimeframeAlignmentStrategy(
        base,
        higher_context,
        TimeframeAlignmentConfig(name="mtf_aligned", context_timeframe="60m", context_column="direction"),
    )

    orders = strategy.generate_orders(pd.DataFrame(), timeframe="15m")
    decisions = strategy.last_filter_decisions.set_index(["filter_name", "order_id"])

    assert orders["order_id"].tolist() == ["accepted"]
    assert ("signal_bar_adapter", "base-rejected") in decisions.index
    assert decisions.loc[("signal_bar_adapter", "base-rejected"), "reason"] == "signal_bar_no_liquidity"
    assert ("higher_timeframe_alignment", "accepted") in decisions.index
    assert decisions.loc[("higher_timeframe_alignment", "accepted"), "status"] == "accepted"


def test_higher_timeframe_filter_decisions_use_vectorized_columns_not_record_loop() -> None:
    source = getsource(_filter_decision_frame)

    assert ".to_dict(" not in source


def test_higher_timeframe_strategy_adds_metadata_without_record_loop() -> None:
    source = getsource(HigherTimeframeAlignmentStrategy.generate_orders)

    assert ".to_dict(" not in source
