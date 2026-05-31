from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from trending_winning.backtest.execution import (
    OrderExecutionResult,
    compute_order_execution_metrics,
    is_favorable_target,
    is_protective_stop,
    normalize_order_side,
)
from trending_winning.backtest.models import ORDER_REQUIRED_COLUMNS
from trending_winning.data.schema import normalize_symbol


def validate_order_frame_columns(orders: pd.DataFrame, *, extra_required: tuple[str, ...] = ()) -> None:
    """校验订单表结构；列缺失属于接入错误，必须用清晰异常暴露。"""
    missing = sorted(ORDER_REQUIRED_COLUMNS.union(extra_required).difference(orders.columns))
    if missing:
        raise ValueError(f"订单缺少必要字段：{', '.join(missing)}")


def order_duplicate_reject_reason(order: Mapping[str, object], seen_order_ids: set[str]) -> str:
    """同一轮回测内订单 ID 必须唯一；空 ID 交给原有撮合规则处理。"""
    order_id = _as_text(order.get("order_id", "")).strip()
    if not order_id:
        return ""
    if order_id in seen_order_ids:
        return "duplicate_order_id"
    seen_order_ids.add(order_id)
    return ""


def order_preflight_reject_reason(order: Mapping[str, object]) -> str:
    """订单进入撮合前的关键字段预检；失败必须进入决策日志。"""
    if not _as_text(order.get("order_id", "")).strip():
        return "invalid_order"
    if not _as_text(order.get("event_id", "")).strip():
        return "invalid_order"
    if not normalize_symbol(order.get("stock_code", "")):
        return "invalid_order"
    side = normalize_order_side(order.get("side", ""))
    if side not in {"long", "short"}:
        return "invalid_order"
    if pd.isna(pd.to_datetime(order.get("signal_date", pd.NaT), errors="coerce")):
        return "invalid_order"
    try:
        signal_index = int(order.get("signal_bar_index", -1))
    except (TypeError, ValueError):
        return "invalid_order"
    if signal_index < 0:
        return "invalid_order"
    if not all(_is_positive_number(order.get(column, None)) for column in ("entry_price", "stop_price", "target_price")):
        return "invalid_order"
    entry_price = _as_float(order.get("entry_price", 0.0))
    stop_price = _as_float(order.get("stop_price", 0.0))
    target_price = _as_float(order.get("target_price", 0.0))
    if not is_protective_stop(side, entry_price, stop_price):
        return "invalid_order"
    if not is_favorable_target(side, entry_price, target_price):
        return "target_not_favorable"
    max_holding = order.get("max_holding_bars", 1)
    if not pd.isna(max_holding) and _as_int(max_holding, default=0) < 1:
        return "invalid_order"
    return ""


def order_decision_record(
    order: pd.Series,
    status: str,
    reason: str,
    *,
    trade: dict[str, object] | None = None,
    capital_fraction: float = 0.0,
    risk_fraction: float = 0.0,
    margin_fraction: float = 0.0,
    sector: str = "",
    portfolio_priority: int | None = None,
    execution: OrderExecutionResult | None = None,
) -> dict[str, object]:
    """生成订单决策日志；记录未成交、被拒绝和接受的统一原因。"""
    source = trade or order
    execution_metrics = _decision_execution_metrics(order, trade, execution)
    side = _decision_side(_field(source, order, "side", ""))
    return {
        "order_id": _field(source, order, "order_id", ""),
        "event_id": _field(source, order, "event_id", ""),
        "event_type": _as_text(_field(source, order, "event_type", "")),
        "strategy_name": _field(source, order, "strategy_name", ""),
        "detector_name": _field(source, order, "detector_name", ""),
        "stock_code": _field(source, order, "stock_code", ""),
        "timeframe": _field(source, order, "timeframe", ""),
        "signal_date": _field(order, source, "signal_date", pd.NaT),
        "signal_bar_index": _as_int(_field(order, source, "signal_bar_index", -1), default=-1),
        "side": side,
        "planned_entry_price": _as_float(
            _field(source, order, "planned_entry_price", _field(order, source, "entry_price", 0.0))
        ),
        "entry_date": _field(source, order, "entry_date", pd.NaT) if trade is not None else pd.NaT,
        **execution_metrics,
        "status": status,
        "reason": reason,
        "portfolio_priority": _as_int(
            portfolio_priority if portfolio_priority is not None else _field(order, source, "_portfolio_priority", 0),
            default=0,
        ),
        "capital_fraction": float(capital_fraction),
        "risk_fraction": float(risk_fraction),
        "margin_fraction": float(margin_fraction),
        "sector": str(sector),
    }


def _decision_execution_metrics(
    order: pd.Series,
    trade: dict[str, object] | None,
    execution: OrderExecutionResult | None,
) -> dict[str, float]:
    if execution is not None:
        return {
            "actual_entry_price": float(execution.actual_entry_price),
            "actual_risk_pct": float(execution.actual_risk_pct),
            "actual_chase_pct": float(execution.actual_chase_pct),
            "actual_reward_to_risk": float(execution.actual_reward_to_risk),
        }
    if trade is None:
        return {
            "actual_entry_price": 0.0,
            "actual_risk_pct": 0.0,
            "actual_chase_pct": 0.0,
            "actual_reward_to_risk": 0.0,
        }
    return compute_order_execution_metrics(
        order,
        str(_field(trade, order, "side", "")),
        _as_float(_field(trade, order, "entry_price", 0.0)),
        _as_float(_field(trade, order, "stop_price", 0.0)),
        _as_float(_field(trade, order, "target_price", 0.0)),
    )


def _decision_side(value: object) -> str:
    normalized = normalize_order_side(value)
    return normalized or _as_text(value)


def _field(primary: object, fallback: object, key: str, default: object) -> object:
    if isinstance(primary, dict) and key in primary:
        return primary[key]
    if hasattr(primary, "get"):
        value = primary.get(key, default)
        if value is not default:
            return value
    if isinstance(fallback, dict) and key in fallback:
        return fallback[key]
    if hasattr(fallback, "get"):
        return fallback.get(key, default)
    return default


def _as_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value)


def _is_positive_number(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _as_int(value: object, *, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default
