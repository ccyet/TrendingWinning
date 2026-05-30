from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from trending_winning.strategies.base import ORDER_COLUMNS, Strategy, empty_orders
from trending_winning.strategies.diagnostics import (
    collect_strategy_filter_decisions,
    empty_strategy_filter_decisions,
    normalize_strategy_filter_decisions,
)


@dataclass(frozen=True)
class StrategyRunResult:
    """单个策略的一次运行结果；订单和过滤日志显式返回，避免依赖策略内部可变状态。"""

    orders: pd.DataFrame = field(default_factory=empty_orders)
    filter_decisions: pd.DataFrame = field(default_factory=empty_strategy_filter_decisions)
    strategy_name: str = ""


@dataclass(frozen=True)
class StrategyBatchRunResult:
    """多个策略的一次运行结果；组合回测只消费这个批量产物，不直接触碰策略内部字段。"""

    orders: pd.DataFrame = field(default_factory=empty_orders)
    filter_decisions: pd.DataFrame = field(default_factory=empty_strategy_filter_decisions)
    runs: tuple[StrategyRunResult, ...] = field(default_factory=tuple)


@runtime_checkable
class StrategyRunProvider(Protocol):
    """显式策略运行协议；新策略优先实现它，旧策略仍可只实现 generate_orders。"""

    name: str

    def generate_order_plan(self, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
        ...


def execute_strategy(strategy: Strategy, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
    """执行单个策略并返回显式产物；兼容旧的 generate_orders + last_filter_decisions 写法。"""
    strategy_name = str(getattr(strategy, "name", ""))
    plan_method = getattr(strategy, "generate_order_plan", None)
    if callable(plan_method):
        return _normalize_strategy_run_result(plan_method(bars, timeframe=timeframe), strategy_name=strategy_name)

    orders = strategy.generate_orders(bars, timeframe=timeframe)
    filters = collect_strategy_filter_decisions([strategy])
    return _normalize_strategy_run_result(
        StrategyRunResult(orders=orders, filter_decisions=filters, strategy_name=strategy_name),
        strategy_name=strategy_name,
    )


def execute_strategies(
    strategies: Sequence[Strategy],
    bars: pd.DataFrame,
    *,
    timeframe: str = "",
) -> StrategyBatchRunResult:
    """批量执行策略；订单和过滤日志同步汇总，保证组合回测和单策略回测使用同一协议。"""
    runs = tuple(execute_strategy(strategy, bars, timeframe=timeframe) for strategy in strategies)
    order_frames = [run.orders for run in runs if not run.orders.empty]
    filter_frames = [run.filter_decisions for run in runs if not run.filter_decisions.empty]
    orders = pd.concat(order_frames, ignore_index=True) if order_frames else empty_orders()
    filters = (
        pd.concat(filter_frames, ignore_index=True)
        if filter_frames
        else empty_strategy_filter_decisions()
    )
    return StrategyBatchRunResult(
        orders=_normalize_order_frame(orders),
        filter_decisions=_sort_strategy_filter_decisions(normalize_strategy_filter_decisions(filters)),
        runs=runs,
    )


def _normalize_strategy_run_result(result: StrategyRunResult, *, strategy_name: str) -> StrategyRunResult:
    """统一策略运行结果字段；这里不做撮合预检，避免策略层和回测层职责混在一起。"""
    normalized_name = result.strategy_name or strategy_name
    orders = _normalize_order_frame(result.orders)
    if normalized_name and not orders.empty:
        orders = orders.copy()
        orders["strategy_name"] = orders["strategy_name"].replace("", normalized_name).fillna(normalized_name)
    return StrategyRunResult(
        orders=orders,
        filter_decisions=_normalize_filter_decision_frame(result.filter_decisions, strategy_name=normalized_name),
        strategy_name=normalized_name,
    )


def _normalize_order_frame(orders: pd.DataFrame) -> pd.DataFrame:
    """补齐策略订单标准字段；订单内容是否合法仍由撮合层统一判定。"""
    if orders.empty:
        return empty_orders()
    result = orders.copy()
    for column in ORDER_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[ORDER_COLUMNS]


def _normalize_filter_decision_frame(decisions: pd.DataFrame, *, strategy_name: str) -> pd.DataFrame:
    """补齐策略过滤日志字段；缺少策略名时用当前运行策略名兜底，便于统计归组。"""
    result = normalize_strategy_filter_decisions(decisions)
    if not strategy_name or result.empty:
        return result
    result = result.copy()
    for column in ("strategy_name", "base_strategy_name"):
        missing = result[column].isna() | result[column].astype("string").str.strip().eq("")
        result.loc[missing.fillna(False), column] = strategy_name
    return result


def _sort_strategy_filter_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    """批量策略过滤日志按全市场信号时间排序，便于和订单决策日志对齐复盘。"""
    if decisions.empty or "signal_date" not in decisions.columns:
        return decisions
    result = decisions.copy()
    result["_signal_date_sort"] = pd.to_datetime(result["signal_date"], errors="coerce")
    result["_signal_bar_sort"] = pd.to_numeric(result["signal_bar_index"], errors="coerce")
    sorted_result = result.sort_values(
        ["_signal_date_sort", "stock_code", "_signal_bar_sort", "order_id"],
        kind="mergesort",
        na_position="last",
    )
    return sorted_result.drop(columns=["_signal_date_sort", "_signal_bar_sort"]).reset_index(drop=True)
