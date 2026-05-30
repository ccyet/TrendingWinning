from __future__ import annotations

import pandas as pd

from trending_winning.strategies.base import ORDER_COLUMNS
from trending_winning.strategies.diagnostics import STRATEGY_FILTER_DECISION_COLUMNS
from trending_winning.strategies.runtime import StrategyRunResult, execute_strategy


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


def test_execute_strategy_prefers_explicit_strategy_run_result() -> None:
    strategy = ExplicitPlanStrategy()

    result = execute_strategy(strategy, pd.DataFrame(), timeframe="15m")

    assert strategy.seen_timeframe == "15m"
    assert result.strategy_name == "explicit_plan"
    assert result.orders["order_id"].tolist() == ["explicit-order"]
    assert result.orders["timeframe"].tolist() == ["15m"]
    assert result.filter_decisions.columns.tolist() == STRATEGY_FILTER_DECISION_COLUMNS
    assert result.filter_decisions.loc[0, "reason"] == "custom_filter"
