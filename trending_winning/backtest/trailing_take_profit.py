from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrailingTakeProfitMasks:
    """回撤止盈向量结果；gap 表示跳空越过回撤线，hit 表示盘中触及回撤线。"""

    gap: np.ndarray
    hit: np.ndarray
    armed: np.ndarray
    prices: np.ndarray


def trailing_take_profit_enabled(activation_pct: float, drawdown_pct: float) -> bool:
    """判断回撤止盈是否启用；任一参数为 0 时保持关闭。"""
    return bool(activation_pct > 0 and drawdown_pct > 0)


def trailing_take_profit_masks(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    liquid: np.ndarray,
    *,
    side: str,
    entry_price: float,
    activation_pct: float,
    drawdown_pct: float,
) -> TrailingTakeProfitMasks:
    """按上一根已完成 K 的持仓峰值/谷值计算回撤止盈。

    多头先等上一根完成 K 的最高价相对实际入场价达到启动浮盈，再用
    `previous_peak * (1 - drawdown_pct)` 作为下一根可用回撤线。
    空头对称使用上一根完成 K 的最低价和 `previous_trough * (1 + drawdown_pct)`。
    """
    length = int(len(opens))
    empty = _empty_masks(length)
    if length == 0 or entry_price <= 0 or not trailing_take_profit_enabled(activation_pct, drawdown_pct):
        return empty
    if not _same_length(length, highs, lows, liquid):
        raise ValueError("回撤止盈输入数组长度必须一致。")

    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    liquid = np.asarray(liquid, dtype=bool)
    side = side.strip().lower()

    if side == "long":
        peak = np.maximum.accumulate(np.where(liquid, highs, entry_price))
        previous_peak = np.concatenate(([entry_price], peak[:-1]))
        prices = np.round(previous_peak * (1.0 - drawdown_pct), 12)
        armed = previous_peak >= entry_price * (1.0 + activation_pct)
        profitable = prices > entry_price
        return TrailingTakeProfitMasks(
            gap=liquid & armed & profitable & (opens <= prices),
            hit=liquid & armed & profitable & (lows <= prices),
            armed=armed & profitable,
            prices=prices,
        )

    if side == "short":
        trough = np.minimum.accumulate(np.where(liquid, lows, entry_price))
        previous_trough = np.concatenate(([entry_price], trough[:-1]))
        prices = np.round(previous_trough * (1.0 + drawdown_pct), 12)
        armed = previous_trough <= entry_price * (1.0 - activation_pct)
        profitable = prices < entry_price
        return TrailingTakeProfitMasks(
            gap=liquid & armed & profitable & (opens >= prices),
            hit=liquid & armed & profitable & (highs >= prices),
            armed=armed & profitable,
            prices=prices,
        )

    return empty


def _empty_masks(length: int) -> TrailingTakeProfitMasks:
    return TrailingTakeProfitMasks(
        gap=np.full(length, False),
        hit=np.full(length, False),
        armed=np.full(length, False),
        prices=np.full(length, np.nan),
    )


def _same_length(length: int, *arrays: np.ndarray) -> bool:
    return all(len(array) == length for array in arrays)
