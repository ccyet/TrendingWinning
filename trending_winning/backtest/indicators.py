from __future__ import annotations

import numpy as np
import pandas as pd


def completed_bar_moving_average(group: pd.DataFrame, path_index: pd.Index, period: int) -> np.ndarray | None:
    """按当前周期收盘价计算上一根完成 K 的均线，并对齐到持仓路径。"""
    if period < 2:
        return None
    closes = pd.to_numeric(group["close"], errors="coerce").astype(float)
    moving_average = closes.rolling(period, min_periods=period).mean().shift(1)
    return moving_average.loc[path_index].to_numpy(dtype=float)
