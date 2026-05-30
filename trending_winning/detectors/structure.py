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
    length = len(result)
    pivot_high, pivot_low = confirmed_pivots(high, low, cfg.left_bars, cfg.right_bars)

    labels = np.full(length, "", dtype=object)
    previous_pivot_high = np.nan
    previous_pivot_low = np.nan
    for index in range(length):
        if pivot_high[index]:
            labels[index] = "HH" if np.isfinite(previous_pivot_high) and high[index] > previous_pivot_high else "LH"
            previous_pivot_high = high[index]
        if pivot_low[index]:
            labels[index] = "HL" if np.isfinite(previous_pivot_low) and low[index] > previous_pivot_low else "LL"
            previous_pivot_low = low[index]
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
    last_high = np.full(length, np.nan)
    last_low = np.full(length, np.nan)
    structure_score = np.zeros(length, dtype=float)
    current_last_high = np.nan
    current_last_low = np.nan
    score = 0.0

    for index in range(length):
        confirmed_index = index - right_bars
        if confirmed_index >= 0:
            if pivot_high[confirmed_index]:
                label = str(labels[confirmed_index])
                score += 1.0 if label == "HH" else -1.0
                current_last_high = high[confirmed_index]
            if pivot_low[confirmed_index]:
                label = str(labels[confirmed_index])
                score += 1.0 if label == "HL" else -1.0
                current_last_low = low[confirmed_index]
        structure_score[index] = score
        last_high[index] = current_last_high
        last_low[index] = current_last_low
    return last_high, last_low, structure_score
