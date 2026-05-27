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
    last_high = np.full(length, np.nan)
    last_low = np.full(length, np.nan)
    previous_pivot_high = np.nan
    previous_pivot_low = np.nan
    current_last_high = np.nan
    current_last_low = np.nan
    structure_score = np.zeros(length, dtype=float)
    score = 0.0

    for index in range(length):
        last_high[index] = current_last_high
        last_low[index] = current_last_low
        if pivot_high[index]:
            labels[index] = "HH" if np.isfinite(previous_pivot_high) and high[index] > previous_pivot_high else "LH"
            score += 1.0 if labels[index] == "HH" else -1.0
            previous_pivot_high = high[index]
            current_last_high = high[index]
        if pivot_low[index]:
            labels[index] = "HL" if np.isfinite(previous_pivot_low) and low[index] > previous_pivot_low else "LL"
            score += 1.0 if labels[index] == "HL" else -1.0
            previous_pivot_low = low[index]
            current_last_low = low[index]
        structure_score[index] = score

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
