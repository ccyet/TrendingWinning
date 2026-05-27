from __future__ import annotations

import numpy as np


def confirmed_pivots(
    high: np.ndarray,
    low: np.ndarray,
    left_bars: int,
    right_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """确认型摆动点；右侧 K 线走完后才标记 pivot high / pivot low。"""
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars 和 right_bars 至少需要 1。")
    if len(high) != len(low):
        raise ValueError("high 和 low 长度必须一致。")

    high_values = np.asarray(high, dtype=float)
    low_values = np.asarray(low, dtype=float)
    length = len(high_values)
    pivot_high = np.full(length, False)
    pivot_low = np.full(length, False)
    window_size = left_bars + right_bars + 1
    if length < window_size:
        return pivot_high, pivot_low

    high_windows = np.lib.stride_tricks.sliding_window_view(high_values, window_size)
    low_windows = np.lib.stride_tricks.sliding_window_view(low_values, window_size)
    target_indexes = np.arange(left_bars, length - right_bars)
    center_high = high_windows[:, left_bars]
    center_low = low_windows[:, left_bars]

    pivot_high[target_indexes] = (center_high == _nanmax_rows(high_windows)) & (
        center_high > _nanmax_rows(high_windows[:, :left_bars])
    )
    pivot_low[target_indexes] = (center_low == _nanmin_rows(low_windows)) & (
        center_low < _nanmin_rows(low_windows[:, :left_bars])
    )
    return pivot_high, pivot_low


def _nanmax_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.max(np.where(finite, values, -np.inf), axis=1)
    out[~finite.any(axis=1)] = np.nan
    return out


def _nanmin_rows(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    out = np.min(np.where(finite, values, np.inf), axis=1)
    out[~finite.any(axis=1)] = np.nan
    return out
