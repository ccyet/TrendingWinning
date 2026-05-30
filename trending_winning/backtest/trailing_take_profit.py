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


def trailing_take_profit_enabled(activation_pct: float, drawdown_pct: float, ma_period: int = 0) -> bool:
    """判断回撤止盈是否启用；启动浮盈必须搭配比例回撤或均线周期。"""
    return bool(activation_pct > 0 and (drawdown_pct > 0 or ma_period >= 2))


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
    moving_average: np.ndarray | None = None,
) -> TrailingTakeProfitMasks:
    """按上一根已完成 K 的持仓峰值/谷值或均线计算回撤止盈。

    多头先等上一根完成 K 的最高价相对实际入场价达到启动浮盈，再用
    `previous_peak * (1 - drawdown_pct)` 作为下一根可用回撤线。
    空头对称使用上一根完成 K 的最低价和 `previous_trough * (1 + drawdown_pct)`。
    如传入 moving_average，它必须已经是上一根已完成 K 的当前周期均线。
    """
    length = int(len(opens))
    empty = _empty_masks(length)
    if length == 0 or entry_price <= 0 or activation_pct <= 0:
        return empty
    if drawdown_pct <= 0 and moving_average is None:
        return empty
    if not _same_length(length, highs, lows, liquid):
        raise ValueError("回撤止盈输入数组长度必须一致。")

    opens = np.asarray(opens, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    liquid = np.asarray(liquid, dtype=bool)
    ma_prices = _moving_average_prices(moving_average, length)
    side = side.strip().lower()

    if side == "long":
        peak = np.maximum.accumulate(np.where(liquid, highs, entry_price))
        previous_peak = np.concatenate(([entry_price], peak[:-1]))
        ratio_prices = _ratio_trailing_prices(previous_peak, drawdown_pct, side="long")
        prices = _long_exit_prices(ratio_prices, ma_prices, entry_price)
        armed = previous_peak >= entry_price * (1.0 + activation_pct)
        active = np.isfinite(prices)
        reported_prices = _reported_prices(prices, ratio_prices, ma_prices)
        return TrailingTakeProfitMasks(
            gap=liquid & armed & active & (opens <= prices),
            hit=liquid & armed & active & (lows <= prices),
            armed=armed & active,
            prices=reported_prices,
        )

    if side == "short":
        trough = np.minimum.accumulate(np.where(liquid, lows, entry_price))
        previous_trough = np.concatenate(([entry_price], trough[:-1]))
        ratio_prices = _ratio_trailing_prices(previous_trough, drawdown_pct, side="short")
        prices = _short_exit_prices(ratio_prices, ma_prices, entry_price)
        armed = previous_trough <= entry_price * (1.0 - activation_pct)
        active = np.isfinite(prices)
        reported_prices = _reported_prices(prices, ratio_prices, ma_prices)
        return TrailingTakeProfitMasks(
            gap=liquid & armed & active & (opens >= prices),
            hit=liquid & armed & active & (highs >= prices),
            armed=armed & active,
            prices=reported_prices,
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


def _moving_average_prices(moving_average: np.ndarray | None, length: int) -> np.ndarray:
    if moving_average is None:
        return np.full(length, np.nan)
    if len(moving_average) != length:
        raise ValueError("回撤止盈均线数组长度必须与 K 线路径一致。")
    return np.round(np.asarray(moving_average, dtype=float), 12)


def _ratio_trailing_prices(extreme: np.ndarray, drawdown_pct: float, *, side: str) -> np.ndarray:
    if drawdown_pct <= 0:
        return np.full(len(extreme), np.nan)
    multiple = 1.0 - drawdown_pct if side == "long" else 1.0 + drawdown_pct
    return np.round(extreme * multiple, 12)


def _long_exit_prices(ratio_prices: np.ndarray, ma_prices: np.ndarray, entry_price: float) -> np.ndarray:
    ratio_active = np.where(ratio_prices > entry_price, ratio_prices, np.nan)
    ma_active = np.where(ma_prices > entry_price, ma_prices, np.nan)
    return np.fmax(ratio_active, ma_active)


def _short_exit_prices(ratio_prices: np.ndarray, ma_prices: np.ndarray, entry_price: float) -> np.ndarray:
    ratio_active = np.where(ratio_prices < entry_price, ratio_prices, np.nan)
    ma_active = np.where(ma_prices < entry_price, ma_prices, np.nan)
    return np.fmin(ratio_active, ma_active)


def _reported_prices(exit_prices: np.ndarray, ratio_prices: np.ndarray, ma_prices: np.ndarray) -> np.ndarray:
    fallback = np.where(np.isfinite(ratio_prices), ratio_prices, ma_prices)
    return np.where(np.isfinite(exit_prices), exit_prices, fallback)
