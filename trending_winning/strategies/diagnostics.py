from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd


STRATEGY_FILTER_DECISION_COLUMNS = [
    "order_id",
    "event_id",
    "strategy_name",
    "base_strategy_name",
    "detector_name",
    "stock_code",
    "timeframe",
    "signal_date",
    "signal_bar_index",
    "side",
    "status",
    "reason",
    "filter_name",
    "context_timeframe",
    "context_date",
    "context_state",
]


@runtime_checkable
class StrategyFilterDecisionProvider(Protocol):
    """策略层过滤日志协议；回测只读取日志，不依赖具体过滤器实现。"""

    last_filter_decisions: pd.DataFrame


def empty_strategy_filter_decisions() -> pd.DataFrame:
    """生成空策略过滤日志表，保持保存和统计时的列结构稳定。"""
    return pd.DataFrame(columns=pd.Index(STRATEGY_FILTER_DECISION_COLUMNS))


def collect_strategy_filter_decisions(strategies: Sequence[object]) -> pd.DataFrame:
    """收集策略对象最近一次生成订单时留下的过滤日志。"""
    frames: list[pd.DataFrame] = []
    for strategy in strategies:
        decisions = getattr(strategy, "last_filter_decisions", None)
        if not isinstance(decisions, pd.DataFrame) or decisions.empty:
            continue
        frames.append(_normalize_strategy_filter_decisions(decisions))
    if not frames:
        return empty_strategy_filter_decisions()
    return pd.concat(frames, ignore_index=True)[STRATEGY_FILTER_DECISION_COLUMNS]


def _normalize_strategy_filter_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    for column in STRATEGY_FILTER_DECISION_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[STRATEGY_FILTER_DECISION_COLUMNS]
