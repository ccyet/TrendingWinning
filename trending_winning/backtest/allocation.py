from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from trending_winning.backtest.execution import normalize_order_side


@dataclass(frozen=True)
class PortfolioAllocationConfig:
    """组合仓位分配参数；capital 是名义仓位，margin 是资金占用。"""

    max_open_positions: int = 5
    capital_per_trade: float | None = None
    risk_per_trade: float | None = None
    max_capital_per_trade: float = 1.0
    short_margin_rate: float = 1.0
    reserve_cash: float = 0.0
    strategy_capital_limit: Mapping[str, float] = field(default_factory=dict)
    sector_capital_limit: Mapping[str, float] = field(default_factory=dict)


def next_capital_fraction(
    open_positions: Sequence[Mapping[str, object]],
    order: Mapping[str, object] | pd.Series,
    *,
    sector: str,
    risk_fraction: float,
    config: PortfolioAllocationConfig,
) -> float:
    """计算下一笔订单可用名义仓位，同时约束现金、保证金、策略和行业额度。"""
    max_capital = 1.0 - config.reserve_cash
    margin_rate = order_margin_rate(order, config)
    used_margin = sum(float(item["margin_fraction"]) for item in open_positions)
    available_margin = max_capital - used_margin
    base = _base_capital_fraction(risk_fraction, max_capital, margin_rate, config)
    strategy_room = _remaining_named_limit(
        open_positions,
        "strategy_name",
        str(_order_value(order, "strategy_name", "")),
        config.strategy_capital_limit,
    )
    sector_room = _remaining_named_limit(open_positions, "sector", sector, config.sector_capital_limit)
    room_by_margin = min(available_margin, strategy_room, sector_room) / margin_rate
    return float(round(max(0.0, min(base, room_by_margin)), 12))


def order_margin_fraction(
    order: Mapping[str, object] | pd.Series,
    capital_fraction: float,
    config: PortfolioAllocationConfig,
) -> float:
    """把名义仓位转换成保证金占用；空头按 short_margin_rate 放大。"""
    return float(capital_fraction * order_margin_rate(order, config))


def order_margin_rate(order: Mapping[str, object] | pd.Series, config: PortfolioAllocationConfig) -> float:
    """返回订单保证金倍率；当前仅空头可高于 1。"""
    return config.short_margin_rate if normalize_order_side(_order_value(order, "side", "long")) == "short" else 1.0


def _base_capital_fraction(
    risk_fraction: float,
    max_capital: float,
    margin_rate: float,
    config: PortfolioAllocationConfig,
) -> float:
    max_trade_notional = config.max_capital_per_trade / margin_rate
    if config.risk_per_trade is not None and risk_fraction > 0:
        return min(config.risk_per_trade / risk_fraction, max_trade_notional)
    if config.capital_per_trade is not None:
        return min(config.capital_per_trade / margin_rate, max_trade_notional)
    return min(max_capital / config.max_open_positions / margin_rate, max_trade_notional)


def _remaining_named_limit(
    open_positions: Sequence[Mapping[str, object]],
    field: str,
    value: str,
    limits: Mapping[str, float],
) -> float:
    if value not in limits:
        return 1.0
    used = sum(float(item["margin_fraction"]) for item in open_positions if str(item.get(field, "")) == value)
    return float(limits[value] - used)


def _order_value(order: Mapping[str, object] | pd.Series, key: str, default: object) -> object:
    return order.get(key, default)
