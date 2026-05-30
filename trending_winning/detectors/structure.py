from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.data.schema import normalize_bars
from trending_winning.detectors.pivots import confirmed_pivots


@dataclass(frozen=True)
class StructureConfig:
    """摆动点结构参数；用左右确认 K 数和突破缓冲定义 HH/HL/BOS。"""

    left_bars: int = 2
    right_bars: int = 2
    break_buffer: float = 0.0


def attach_market_structure(bars: pd.DataFrame, config: StructureConfig | None = None) -> pd.DataFrame:
    cfg = config or StructureConfig()
    if cfg.left_bars < 1 or cfg.right_bars < 1:
        raise ValueError("left_bars 和 right_bars 至少需要 1。")
    if cfg.break_buffer < 0:
        raise ValueError("break_buffer 不能为负数。")

    normalized = normalize_bars(bars)
    if normalized.empty:
        return _empty_structure(normalized)

    frames: list[pd.DataFrame] = []
    for _, group in normalized.groupby("stock_code", sort=False):
        frames.append(_attach_group_structure(group.reset_index(drop=True), cfg))
    return pd.concat(frames, ignore_index=True)


def _empty_structure(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        "pivot_high",
        "pivot_low",
        "last_swing_high",
        "last_swing_low",
        "bos_up",
        "bos_down",
        "choch_up",
        "choch_down",
        "structure_score",
    ]:
        result[column] = pd.Series(dtype=float)
    result["structure_label"] = pd.Series(dtype=str)
    return result


def _attach_group_structure(group: pd.DataFrame, cfg: StructureConfig) -> pd.DataFrame:
    result = group.copy()
    high = result["high"].astype(float).to_numpy()
    low = result["low"].astype(float).to_numpy()
    close = result["close"].astype(float).to_numpy()
    pivot_high, pivot_low = confirmed_pivots(high, low, cfg.left_bars, cfg.right_bars)

    labels = _structure_labels(high, low, pivot_high, pivot_low)
    last_high, last_low, structure_score = _confirmed_structure_state(
        high,
        low,
        pivot_high,
        pivot_low,
        labels,
        right_bars=cfg.right_bars,
    )

    bos_up = close > last_high * (1.0 + cfg.break_buffer)
    bos_down = close < last_low * (1.0 - cfg.break_buffer)
    result["pivot_high"] = pivot_high
    result["pivot_low"] = pivot_low
    result["last_swing_high"] = last_high
    result["last_swing_low"] = last_low
    result["structure_label"] = labels
    result["bos_up"] = np.where(np.isfinite(last_high), bos_up, False)
    result["bos_down"] = np.where(np.isfinite(last_low), bos_down, False)
    result["choch_up"] = result["bos_up"] & (pd.Series(structure_score).shift(1).fillna(0.0) < 0).to_numpy()
    result["choch_down"] = result["bos_down"] & (pd.Series(structure_score).shift(1).fillna(0.0) > 0).to_numpy()
    result["structure_score"] = structure_score
    return result


def _structure_labels(
    high: np.ndarray,
    low: np.ndarray,
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
) -> np.ndarray:
    """按已确认 pivot 序列批量标记 HH/HL/LH/LL。"""
    labels = np.full(len(high), "", dtype=object)
    high_indexes = np.flatnonzero(pivot_high)
    if len(high_indexes) > 0:
        high_values = high[high_indexes]
        previous_high = np.concatenate(([np.nan], high_values[:-1]))
        labels[high_indexes] = np.where(np.isfinite(previous_high) & (high_values > previous_high), "HH", "LH")

    low_indexes = np.flatnonzero(pivot_low)
    if len(low_indexes) > 0:
        low_values = low[low_indexes]
        previous_low = np.concatenate(([np.nan], low_values[:-1]))
        labels[low_indexes] = np.where(np.isfinite(previous_low) & (low_values > previous_low), "HL", "LL")
    return labels


def _confirmed_structure_state(
    high: np.ndarray,
    low: np.ndarray,
    pivot_high: np.ndarray,
    pivot_low: np.ndarray,
    labels: np.ndarray,
    *,
    right_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把原始 pivot 延迟到确认 K 线完成后，避免结构字段提前暴露未来信息。"""
    length = len(high)
    score_delta = np.zeros(length, dtype=float)
    high_updates = np.full(length, np.nan)
    low_updates = np.full(length, np.nan)

    high_pivot_indexes = np.flatnonzero(pivot_high)
    high_confirm_indexes = high_pivot_indexes + right_bars
    high_valid = high_confirm_indexes < length
    high_pivot_indexes = high_pivot_indexes[high_valid]
    high_confirm_indexes = high_confirm_indexes[high_valid]
    if len(high_confirm_indexes) > 0:
        high_labels = labels[high_pivot_indexes]
        high_delta = np.where(high_labels == "HH", 1.0, -1.0)
        np.add.at(score_delta, high_confirm_indexes, high_delta)
        high_updates[high_confirm_indexes] = high[high_pivot_indexes]

    low_pivot_indexes = np.flatnonzero(pivot_low)
    low_confirm_indexes = low_pivot_indexes + right_bars
    low_valid = low_confirm_indexes < length
    low_pivot_indexes = low_pivot_indexes[low_valid]
    low_confirm_indexes = low_confirm_indexes[low_valid]
    if len(low_confirm_indexes) > 0:
        low_labels = labels[low_pivot_indexes]
        low_delta = np.where(low_labels == "HL", 1.0, -1.0)
        np.add.at(score_delta, low_confirm_indexes, low_delta)
        low_updates[low_confirm_indexes] = low[low_pivot_indexes]

    return _forward_fill_float(high_updates), _forward_fill_float(low_updates), np.cumsum(score_delta)


def _forward_fill_float(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, dtype="float64").ffill().to_numpy(dtype=float)
