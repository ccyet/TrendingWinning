from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trending_winning.backtest.trailing_take_profit import (
    trailing_take_profit_masks as compute_trailing_take_profit_masks,
)
from trending_winning.backtest.indicators import completed_bar_moving_average


@dataclass(frozen=True)
class OrderExecutionResult:
    """订单撮合结果；兼容旧二元解包，并携带真实成交后的风控诊断。"""

    trade: dict[str, object] | None = None
    reject_reason: str = ""
    actual_entry_price: float = 0.0
    actual_risk_pct: float = 0.0
    actual_chase_pct: float = 0.0
    actual_reward_to_risk: float = 0.0

    def __iter__(self) -> Iterator[object]:
        yield self.trade
        yield self.reject_reason


def validate_backtest_config(cfg: Any) -> None:
    """校验回测撮合参数；供单策略和组合策略共用。"""
    if cfg.take_profit_pct < 0 or cfg.stop_loss_pct < 0:
        raise ValueError("take_profit_pct 和 stop_loss_pct 不能为负数。")
    if cfg.max_holding_bars < 1:
        raise ValueError("max_holding_bars 至少需要 1。")
    if cfg.fee_rate < 0 or cfg.slippage_bps < 0:
        raise ValueError("fee_rate 和 slippage_bps 不能为负数。")
    if cfg.initial_equity <= 0:
        raise ValueError("initial_equity 必须大于 0。")
    if getattr(cfg, "intrabar_exit_policy", "conservative") not in {"conservative", "optimistic"}:
        raise ValueError("intrabar_exit_policy 仅支持 conservative 或 optimistic。")
    activation_pct = float(getattr(cfg, "trailing_take_profit_activation_pct", 0.0))
    drawdown_pct = float(getattr(cfg, "trailing_take_profit_drawdown_pct", 0.0))
    ma_period = int(getattr(cfg, "trailing_take_profit_ma_period", 0))
    if activation_pct < 0 or not 0 <= drawdown_pct < 1 or ma_period < 0:
        raise ValueError("trailing_take_profit 参数必须非负，且回撤幅度必须小于 1。")
    if ma_period == 1:
        raise ValueError("trailing_take_profit 均线周期只能为 0 或至少 2。")
    if activation_pct > 0 and drawdown_pct == 0 and ma_period == 0:
        raise ValueError("trailing_take_profit 启动浮盈必须与比例回撤或均线周期同时启用。")


def simulate_order_trade(
    group: pd.DataFrame,
    order: pd.Series,
    signal_index: int,
    cfg: Any,
) -> dict[str, object] | None:
    """按信号 K 后一根触发价撮合订单，并返回逐笔交易记录。"""
    trade, _reason = simulate_order_trade_with_rejection(group, order, signal_index, cfg)
    return trade


def simulate_order_trade_with_rejection(
    group: pd.DataFrame,
    order: pd.Series,
    signal_index: int,
    cfg: Any,
) -> OrderExecutionResult:
    """撮合订单并返回拒绝原因和真实成交诊断；供订单决策日志使用。"""
    validate_backtest_config(cfg)
    side = normalize_order_side(order["side"])
    if side not in {"long", "short"}:
        return OrderExecutionResult(reject_reason="invalid_order")
    entry_price = float(order["entry_price"])
    stop_price = float(order["stop_price"])
    target_price = float(order["target_price"])
    max_holding = _order_max_holding_bars(order, cfg)
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0 or max_holding < 1:
        return OrderExecutionResult(reject_reason="invalid_order")
    if not is_protective_stop(side, entry_price, stop_price):
        return OrderExecutionResult(reject_reason="invalid_order")
    if not is_favorable_target(side, entry_price, target_price):
        return OrderExecutionResult(reject_reason="target_not_favorable")
    if signal_index < 0:
        return OrderExecutionResult(reject_reason="invalid_order")
    if signal_index >= len(group):
        return OrderExecutionResult(reject_reason="no_bars")

    entry_index = signal_index + 1
    if entry_index > len(group) - 1:
        return OrderExecutionResult(reject_reason="no_fill")
    entry_row = group.loc[entry_index]
    if not _is_liquid_bar(entry_row):
        return OrderExecutionResult(reject_reason="no_liquidity")
    if side == "long" and float(entry_row["high"]) < entry_price:
        return OrderExecutionResult(reject_reason="no_fill")
    if side == "short" and float(entry_row["low"]) > entry_price:
        return OrderExecutionResult(reject_reason="no_fill")
    fill_price = _stop_entry_fill_price(entry_row, side, entry_price)

    direction = 1.0 if side == "long" else -1.0
    slipped_entry = apply_slippage(fill_price, direction, cfg)
    diagnostics = compute_order_execution_metrics(order, side, slipped_entry, stop_price, target_price)
    rejection = _entry_constraint_rejection(order, side, slipped_entry, stop_price, target_price)
    if rejection:
        return OrderExecutionResult(trade=None, reject_reason=rejection, **diagnostics)
    last_exit_index = min(len(group) - 1, entry_index + max_holding)
    exit_index, exit_price, exit_reason = _first_exit_after_entry(
        group,
        side=side,
        entry_index=entry_index,
        last_exit_index=last_exit_index,
        entry_price=slipped_entry,
        stop_price=stop_price,
        target_price=target_price,
        policy=str(getattr(cfg, "intrabar_exit_policy", "conservative")),
        trailing_activation_pct=float(getattr(cfg, "trailing_take_profit_activation_pct", 0.0)),
        trailing_drawdown_pct=float(getattr(cfg, "trailing_take_profit_drawdown_pct", 0.0)),
        trailing_ma_period=int(getattr(cfg, "trailing_take_profit_ma_period", 0)),
    )
    slipped_exit = apply_slippage(exit_price, -direction, cfg)
    gross_return = direction * (slipped_exit / slipped_entry - 1.0)
    net_return = gross_return - 2.0 * cfg.fee_rate
    return_pct = float(net_return * 100.0)
    path_metrics = trade_path_metrics(
        group,
        side=side,
        entry_index=entry_index,
        exit_index=exit_index,
        entry_price=slipped_entry,
        stop_price=stop_price,
        return_pct=return_pct,
    )
    trade = {
        "order_id": order.get("order_id", ""),
        "event_id": order.get("event_id", ""),
        "event_type": _safe_text(order.get("event_type", "")),
        "strategy_name": order.get("strategy_name", ""),
        "detector_name": order.get("detector_name", ""),
        "stock_code": order["stock_code"],
        "timeframe": order.get("timeframe", ""),
        "signal_date": order.get("signal_date", group.loc[signal_index, "date"]),
        "signal_bar_index": int(signal_index),
        "side": side,
        "planned_entry_price": entry_price,
        "entry_date": group.loc[entry_index, "date"],
        "entry_price": slipped_entry,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_share": float(abs(slipped_entry - stop_price)),
        "exit_date": group.loc[exit_index, "date"],
        "exit_price": slipped_exit,
        "exit_reason": exit_reason,
        "holding_bars": int(exit_index - entry_index),
        "return_pct": return_pct,
        **path_metrics,
        "metadata": order.get("metadata", {}),
        "_exit_index": int(exit_index),
    }
    return OrderExecutionResult(trade=trade, reject_reason="", **diagnostics)


def coerce_order_execution_result(value: object, *, order: pd.Series | None = None) -> OrderExecutionResult:
    """把旧 tuple 返回值转成结构化结果；便于测试替身和旧调用继续工作。"""
    if isinstance(value, OrderExecutionResult):
        return value
    if isinstance(value, Sequence) and len(value) >= 2:
        trade = value[0]
        reject_reason = str(value[1] or "")
        metrics = _empty_execution_metrics()
        if isinstance(trade, dict) and order is not None:
            trade = {**trade, "event_type": _safe_text(trade.get("event_type", order.get("event_type", "")))}
            metrics = compute_order_execution_metrics(
                order,
                str(trade.get("side", order.get("side", ""))),
                _safe_float(trade.get("entry_price", 0.0)),
                _safe_float(trade.get("stop_price", order.get("stop_price", 0.0))),
                _safe_float(trade.get("target_price", order.get("target_price", 0.0))),
            )
        return OrderExecutionResult(trade=trade if isinstance(trade, dict) else None, reject_reason=reject_reason, **metrics)
    raise TypeError("撮合结果必须是 OrderExecutionResult 或二元 tuple。")


def compute_order_execution_metrics(
    order: pd.Series,
    side: str,
    actual_entry_price: float,
    stop_price: float,
    target_price: float,
) -> dict[str, float]:
    """计算真实入场后的风险、追价和盈亏比，所有值均用小数比例表达。"""
    if actual_entry_price <= 0:
        return _empty_execution_metrics()
    normalized_side = normalize_order_side(side)
    return {
        "actual_entry_price": _round_float(actual_entry_price),
        "actual_risk_pct": _round_float(_actual_risk_pct(actual_entry_price, stop_price)),
        "actual_chase_pct": _round_float(_actual_chase_pct(order, actual_entry_price)),
        "actual_reward_to_risk": _round_float(
            _actual_reward_to_risk(normalized_side, actual_entry_price, stop_price, target_price)
        ),
    }


def _empty_execution_metrics() -> dict[str, float]:
    return {
        "actual_entry_price": 0.0,
        "actual_risk_pct": 0.0,
        "actual_chase_pct": 0.0,
        "actual_reward_to_risk": 0.0,
    }


def _order_max_holding_bars(order: pd.Series, cfg: Any) -> int:
    """解析订单持仓周期；缺失值使用配置默认值，非法值交给预检或撮合拒单。"""
    value = order.get("max_holding_bars", cfg.max_holding_bars)
    try:
        if pd.isna(value):
            value = cfg.max_holding_bars
        return int(value)
    except (TypeError, ValueError):
        return 0


def apply_slippage(price: float, direction: float, cfg: Any) -> float:
    return float(price * (1.0 + direction * cfg.slippage_bps / 10000.0))


def _first_exit_after_entry(
    group: pd.DataFrame,
    *,
    side: str,
    entry_index: int,
    last_exit_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    policy: str,
    trailing_activation_pct: float,
    trailing_drawdown_pct: float,
    trailing_ma_period: int,
) -> tuple[int, float, str]:
    """向量化查找首个退出 K；优先级保持 gap、同 K 冲突、普通 stop/target。"""
    path = group.loc[entry_index:last_exit_index]
    opens = pd.to_numeric(path["open"], errors="coerce").astype(float).to_numpy()
    highs = pd.to_numeric(path["high"], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(path["low"], errors="coerce").astype(float).to_numpy()
    liquid = _liquid_bar_mask(path)
    reasons = np.full(len(path), "", dtype=object)
    prices = np.full(len(path), np.nan)

    if side == "long":
        gap_stop = liquid & (opens <= stop_price)
        gap_target = liquid & (opens >= target_price)
        hit_stop = liquid & (lows <= stop_price)
        hit_target = liquid & (highs >= target_price)
    else:
        gap_stop = liquid & (opens >= stop_price)
        gap_target = liquid & (opens <= target_price)
        hit_stop = liquid & (highs >= stop_price)
        hit_target = liquid & (lows <= target_price)
    gap_trailing, hit_trailing, trailing_prices = _trailing_take_profit_masks(
        opens,
        highs,
        lows,
        liquid,
        side=side,
        entry_price=entry_price,
        activation_pct=trailing_activation_pct,
        drawdown_pct=trailing_drawdown_pct,
        moving_average=completed_bar_moving_average(group, path.index, trailing_ma_period),
    )

    _set_exit_values(reasons, prices, gap_stop, opens, "stop_loss")
    _set_exit_values(reasons, prices, (reasons == "") & gap_target, opens, "take_profit")
    _set_exit_values(reasons, prices, (reasons == "") & gap_trailing, opens, "trailing_take_profit")

    conflict = (reasons == "") & hit_stop & hit_target
    if policy == "optimistic":
        _set_exit_values(reasons, prices, conflict, target_price, "take_profit")
    else:
        _set_exit_values(reasons, prices, conflict, stop_price, "stop_loss")

    stop_trailing_conflict = (reasons == "") & hit_stop & hit_trailing
    if policy == "optimistic":
        _set_exit_values(reasons, prices, stop_trailing_conflict, trailing_prices, "trailing_take_profit")
    else:
        _set_exit_values(reasons, prices, stop_trailing_conflict, stop_price, "stop_loss")

    target_trailing_conflict = (reasons == "") & hit_target & hit_trailing
    if policy == "optimistic":
        _set_exit_values(reasons, prices, target_trailing_conflict, target_price, "take_profit")
    else:
        _set_exit_values(reasons, prices, target_trailing_conflict, trailing_prices, "trailing_take_profit")

    _set_exit_values(reasons, prices, (reasons == "") & hit_target, target_price, "take_profit")
    _set_exit_values(
        reasons,
        prices,
        (reasons == "") & hit_trailing,
        trailing_prices,
        "trailing_take_profit",
    )
    _set_exit_values(reasons, prices, (reasons == "") & hit_stop, stop_price, "stop_loss")

    hit_positions = np.flatnonzero(reasons != "")
    if len(hit_positions) == 0:
        liquid_positions = np.flatnonzero(liquid)
        if len(liquid_positions) == 0:
            return last_exit_index, float(group.loc[last_exit_index, "close"]), "max_holding"
        last_liquid_position = int(liquid_positions[-1])
        last_liquid_index = int(path.index[last_liquid_position])
        return last_liquid_index, float(group.loc[last_liquid_index, "close"]), "max_holding"
    first = int(hit_positions[0])
    return int(path.index[first]), float(prices[first]), str(reasons[first])


def _trailing_take_profit_masks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    liquid: np.ndarray,
    *,
    side: str,
    entry_price: float,
    activation_pct: float,
    drawdown_pct: float,
    moving_average: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """兼容旧调用签名，实际计算下沉到独立回撤止盈模块。"""
    result = compute_trailing_take_profit_masks(
        opens,
        highs,
        lows,
        liquid,
        side=side,
        entry_price=entry_price,
        activation_pct=activation_pct,
        drawdown_pct=drawdown_pct,
        moving_average=moving_average,
    )
    return result.gap, result.hit, result.prices


def _set_exit_values(
    reasons: np.ndarray,
    prices: np.ndarray,
    mask: np.ndarray,
    price: np.ndarray | float,
    reason: str,
) -> None:
    reasons[mask] = reason
    prices[mask] = price[mask] if isinstance(price, np.ndarray) else float(price)


def trade_path_metrics(
    group: pd.DataFrame,
    *,
    side: str,
    entry_index: int,
    exit_index: int,
    entry_price: float,
    stop_price: float,
    return_pct: float,
) -> dict[str, float]:
    """统计持仓路径的 R 倍数和最大有利/不利波动，供所有回测入口复用。"""
    if entry_price <= 0:
        return _empty_path_metrics()
    path = group.loc[entry_index:exit_index]
    if path.empty:
        return _empty_path_metrics()
    path = path.loc[_liquid_bar_mask(path)]
    if path.empty:
        return _empty_path_metrics()
    high = pd.to_numeric(path["high"], errors="coerce").max()
    low = pd.to_numeric(path["low"], errors="coerce").min()
    if pd.isna(high) or pd.isna(low):
        return _empty_path_metrics()

    direction = 1.0 if side == "long" else -1.0
    favorable_price = float(high) if side == "long" else float(low)
    adverse_price = float(low) if side == "long" else float(high)
    risk_per_share = abs(entry_price - stop_price)
    mfe_pct = max(0.0, direction * (favorable_price / entry_price - 1.0) * 100.0)
    mae_pct = min(0.0, direction * (adverse_price / entry_price - 1.0) * 100.0)
    return {
        "r_multiple": _ratio_pct_to_r(return_pct, entry_price, risk_per_share),
        "mae_pct": _round_float(mae_pct),
        "mfe_pct": _round_float(mfe_pct),
        "mae_r": _ratio_pct_to_r(mae_pct, entry_price, risk_per_share),
        "mfe_r": _ratio_pct_to_r(mfe_pct, entry_price, risk_per_share),
    }


def _empty_path_metrics() -> dict[str, float]:
    return {"r_multiple": 0.0, "mae_pct": 0.0, "mfe_pct": 0.0, "mae_r": 0.0, "mfe_r": 0.0}


def _ratio_pct_to_r(value_pct: float, entry_price: float, risk_per_share: float) -> float:
    if entry_price <= 0 or risk_per_share <= 0:
        return 0.0
    return _round_float(float(value_pct) / 100.0 * entry_price / risk_per_share)


def _round_float(value: float) -> float:
    return float(round(float(value), 12))


def _stop_entry_fill_price(row: pd.Series, side: str, stop_price: float) -> float:
    open_price = float(row["open"])
    if side == "long" and open_price > stop_price:
        return open_price
    if side == "short" and open_price < stop_price:
        return open_price
    return float(stop_price)


def _is_liquid_bar(row: pd.Series) -> bool:
    """判断单根 K 是否可成交；TDX 停牌或无成交分钟 K 不参与真实撮合。"""
    if "volume" not in row.index or "amount" not in row.index:
        return True
    return _positive_float_or_none(row.get("volume")) is not None and _positive_float_or_none(row.get("amount")) is not None


def liquid_bar_mask(frame: pd.DataFrame) -> np.ndarray:
    """返回可成交 K 的布尔掩码；缺少量额字段时兼容旧数据全部视为可成交。"""
    if "volume" not in frame.columns or "amount" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).to_numpy()
    amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0).to_numpy()
    return (volume > 0.0) & (amount > 0.0)


def _liquid_bar_mask(frame: pd.DataFrame) -> np.ndarray:
    return liquid_bar_mask(frame)


def _entry_constraint_rejection(
    order: pd.Series,
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> str:
    max_actual_risk_pct = _positive_float_or_none(order.get("max_actual_risk_pct", None))
    if max_actual_risk_pct is not None and _actual_risk_pct(entry_price, stop_price) > max_actual_risk_pct:
        return "actual_risk_too_high"
    max_chase_pct = _positive_float_or_none(order.get("max_chase_pct", None))
    signal_price = _positive_float_or_none(order.get("signal_price", None))
    if max_chase_pct is not None and signal_price is not None:
        chase_pct = abs(entry_price - signal_price) / signal_price
        if chase_pct > max_chase_pct:
            return "chase_too_far"
    if not is_favorable_target(side, entry_price, target_price):
        return "target_not_favorable"
    return ""


def is_favorable_target(side: str, entry_price: float, target_price: float) -> bool:
    """校验目标价方向：多头目标在入场上方，空头目标在入场下方。"""
    side = normalize_order_side(side)
    if side == "long":
        return target_price > entry_price
    if side == "short":
        return target_price < entry_price
    return False


def is_protective_stop(side: str, entry_price: float, stop_price: float) -> bool:
    """校验止损方向：多头止损在入场下方，空头止损在入场上方。"""
    side = normalize_order_side(side)
    if side == "long":
        return stop_price < entry_price
    if side == "short":
        return stop_price > entry_price
    return False


def normalize_order_side(value: object) -> str:
    """把外部策略传入的方向文本标准化成 long/short，无法识别时返回空串。"""
    text = _safe_text(value).strip().lower()
    return text if text in {"long", "short"} else ""


def _actual_risk_pct(entry_price: float, stop_price: float) -> float:
    if entry_price <= 0:
        return float("inf")
    return abs(entry_price - stop_price) / entry_price


def _actual_chase_pct(order: pd.Series, entry_price: float) -> float:
    signal_price = _positive_float_or_none(order.get("signal_price", None))
    if signal_price is None or entry_price <= 0:
        return 0.0
    return abs(entry_price - signal_price) / signal_price


def _actual_reward_to_risk(side: str, entry_price: float, stop_price: float, target_price: float) -> float:
    risk_per_share = abs(entry_price - stop_price)
    if entry_price <= 0 or risk_per_share <= 0:
        return 0.0
    direction = 1.0 if side == "long" else -1.0
    return direction * (target_price - entry_price) / risk_per_share


def _positive_float_or_none(value: object) -> float | None:
    try:
        if pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value)
