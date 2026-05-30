from __future__ import annotations

import pandas as pd

from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.diagnostics import STRATEGY_FILTER_DECISION_COLUMNS
from trending_winning.strategies.runtime import StrategyRunResult, execute_strategies, execute_strategy


class ExplicitPlanStrategy:
    name = "explicit_plan"

    def __init__(self) -> None:
        self.seen_timeframe = ""

    def generate_order_plan(self, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
        self.seen_timeframe = timeframe
        orders = pd.DataFrame(
            [
                {
                    "order_id": "explicit-order",
                    "strategy_name": self.name,
                    "detector_name": "trend",
                    "event_id": "event:explicit-order",
                    "event_type": "bull_h2_setup",
                    "stock_code": "000001.SZ",
                    "timeframe": timeframe,
                    "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                    "signal_bar_index": 0,
                    "side": "long",
                    "signal_price": 10.0,
                    "entry_price": 10.2,
                    "stop_price": 9.8,
                    "target_price": 11.0,
                    "max_holding_bars": 3,
                    "max_actual_risk_pct": None,
                    "max_chase_pct": None,
                    "metadata": {},
                }
            ],
            columns=ORDER_COLUMNS,
        )
        filters = pd.DataFrame(
            [
                {
                    "order_id": "explicit-filter",
                    "event_id": "event:explicit-filter",
                    "strategy_name": self.name,
                    "detector_name": "trend",
                    "status": "rejected",
                    "reason": "custom_filter",
                    "filter_name": "pure_strategy",
                }
            ]
        )
        return StrategyRunResult(orders=orders, filter_decisions=filters)

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        raise AssertionError("显式策略运行结果可用时，不应回退到 generate_orders。")


class FilterOnlyStrategy:
    def __init__(self, name: str, *, order_id: str, signal_date: str) -> None:
        self.name = name
        self._order_id = order_id
        self._signal_date = pd.Timestamp(signal_date)

    def generate_order_plan(self, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
        filters = pd.DataFrame(
            [
                {
                    "order_id": self._order_id,
                    "event_id": f"event:{self._order_id}",
                    "strategy_name": self.name,
                    "detector_name": "trend",
                    "event_type": "bull_h2_setup",
                    "stock_code": "000001.SZ",
                    "timeframe": timeframe,
                    "signal_date": self._signal_date,
                    "signal_bar_index": 0,
                    "side": "long",
                    "status": "rejected",
                    "reason": "custom_filter",
                    "filter_name": "pure_strategy",
                }
            ]
        )
        return StrategyRunResult(filter_decisions=filters)

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        raise AssertionError("显式策略运行结果可用时，不应回退到 generate_orders。")


def test_execute_strategy_prefers_explicit_strategy_run_result() -> None:
    strategy = ExplicitPlanStrategy()

    result = execute_strategy(strategy, pd.DataFrame(), timeframe="15m")

    assert strategy.seen_timeframe == "15m"
    assert result.strategy_name == "explicit_plan"
    assert result.orders["order_id"].tolist() == ["explicit-order"]
    assert result.orders["timeframe"].tolist() == ["15m"]
    assert result.filter_decisions.columns.tolist() == STRATEGY_FILTER_DECISION_COLUMNS
    assert result.filter_decisions.loc[0, "reason"] == "custom_filter"


def test_execute_strategies_orders_filter_decisions_by_signal_time_across_strategies() -> None:
    late_strategy = FilterOnlyStrategy("late_filter", order_id="late-filter", signal_date="2026-05-25 10:30:00")
    early_strategy = FilterOnlyStrategy("early_filter", order_id="early-filter", signal_date="2026-05-25 09:30:00")

    result = execute_strategies([late_strategy, early_strategy], pd.DataFrame(), timeframe="15m")

    assert result.filter_decisions["order_id"].tolist() == ["early-filter", "late-filter"]
    assert result.filter_decisions["signal_date"].tolist() == sorted(result.filter_decisions["signal_date"].tolist())
