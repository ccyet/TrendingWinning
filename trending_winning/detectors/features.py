from __future__ import annotations

import numpy as np
import pandas as pd

from trending_winning.data.schema import normalize_bars


def attach_bar_features(bars: pd.DataFrame, *, ema_fast: int = 20, ema_slow: int = 50, atr_lookback: int = 14) -> pd.DataFrame:
    if ema_fast < 2 or ema_slow < 2 or atr_lookback < 2:
        raise ValueError("ema_fast、ema_slow、atr_lookback 都至少需要 2。")

    result = normalize_bars(bars)
    if result.empty:
        return result

    frames: list[pd.DataFrame] = []
    for _, group in result.groupby("stock_code", sort=False):
        frames.append(_attach_group_features(group.reset_index(drop=True), ema_fast, ema_slow, atr_lookback))
    return pd.concat(frames, ignore_index=True)


def _attach_group_features(group: pd.DataFrame, ema_fast: int, ema_slow: int, atr_lookback: int) -> pd.DataFrame:
    result = group.copy()
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    open_ = result["open"].astype(float)
    close = result["close"].astype(float)
    prev_close = close.shift(1)
    candle_range = high - low
    body = (close - open_).abs()
    upper_tail = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_tail = pd.concat([open_, close], axis=1).min(axis=1) - low
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    result["bar_range"] = candle_range
    result["bar_body"] = body
    result["close_pos"] = ((close - low) / candle_range.replace(0, np.nan)).fillna(0.5)
    result["body_ratio"] = (body / candle_range.replace(0, np.nan)).fillna(0.0)
    result["upper_tail"] = upper_tail
    result["lower_tail"] = lower_tail
    result["tail_ratio"] = ((upper_tail + lower_tail) / candle_range.replace(0, np.nan)).fillna(0.0)
    result["atr"] = true_range.rolling(atr_lookback, min_periods=2).mean()
    result["ema_fast"] = close.ewm(span=ema_fast, adjust=False, min_periods=2).mean()
    result["ema_slow"] = close.ewm(span=ema_slow, adjust=False, min_periods=2).mean()
    result["ma_align"] = np.select(
        [close > result["ema_fast"], close < result["ema_fast"]],
        [1.0, -1.0],
        default=0.0,
    )
    if ema_slow <= len(result):
        result.loc[close > result["ema_slow"], "ma_align"] += 0.5
        result.loc[close < result["ema_slow"], "ma_align"] -= 0.5
    result["follow_up"] = close > high.shift(1)
    result["follow_down"] = close < low.shift(1)
    return result


def rolling_slope_z(close: pd.Series, lookback: int) -> pd.Series:
    if lookback < 3:
        raise ValueError("lookback 至少需要 3。")

    values = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    log_values = np.log(np.where(values > 0, values, np.nan))
    returns = pd.Series(log_values).diff().rolling(lookback, min_periods=3).std().to_numpy()
    if len(values) < lookback:
        return pd.Series(out, index=close.index)

    finite = np.isfinite(log_values)
    filled_logs = np.where(finite, log_values, 0.0)
    positions = np.arange(len(values), dtype=float)
    prefix_logs = np.r_[0.0, np.cumsum(filled_logs)]
    prefix_weighted_logs = np.r_[0.0, np.cumsum(filled_logs * positions)]
    prefix_valid = np.r_[0, np.cumsum(finite.astype(int))]

    ends = np.arange(lookback - 1, len(values))
    starts = ends - lookback + 1
    local_mean_x = float((lookback - 1) / 2.0)
    x_centered = np.arange(lookback, dtype=float) - local_mean_x
    denominator = float(np.dot(x_centered, x_centered))

    sum_logs = prefix_logs[ends + 1] - prefix_logs[starts]
    sum_weighted_logs = prefix_weighted_logs[ends + 1] - prefix_weighted_logs[starts]
    valid_windows = (prefix_valid[ends + 1] - prefix_valid[starts]) == lookback

    # 将局部窗口 k*y 的回归项转成全局 index*y 的前缀和，避免逐窗口回看。
    slopes = (sum_weighted_logs - (starts.astype(float) + local_mean_x) * sum_logs) / denominator
    volatility = returns[ends]
    slope_z = np.zeros(len(ends), dtype=float)
    non_zero_volatility = np.isfinite(volatility) & (volatility != 0)
    slope_z[non_zero_volatility] = slopes[non_zero_volatility] / volatility[non_zero_volatility]
    out[ends[valid_windows]] = slope_z[valid_windows]
    return pd.Series(out, index=close.index)
