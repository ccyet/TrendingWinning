from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelConfig:
    """旧版通道扫描参数；保留给原有 scan_bars 流程兼容使用。"""

    lookback: int = 40
    min_slope: float = 0.0
    band_atr_multiple: float = 1.0


def attach_trend_channel(bars: pd.DataFrame, config: ChannelConfig | None = None) -> pd.DataFrame:
    cfg = config or ChannelConfig()
    if cfg.lookback < 3:
        raise ValueError("lookback 至少需要 3。")
    if cfg.band_atr_multiple < 0:
        raise ValueError("band_atr_multiple 不能为负数。")

    frames: list[pd.DataFrame] = []
    for _, group in bars.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False):
        frames.append(_attach_group_channel(group.reset_index(drop=True), cfg))
    if frames:
        return pd.concat(frames, ignore_index=True)

    result = bars.copy()
    result["channel_slope"] = pd.Series(dtype=float)
    result["channel_mid"] = pd.Series(dtype=float)
    result["channel_upper"] = pd.Series(dtype=float)
    result["channel_lower"] = pd.Series(dtype=float)
    result["channel_width"] = pd.Series(dtype=float)
    result["channel_direction"] = pd.Series(dtype=str)
    return result


def _attach_group_channel(group: pd.DataFrame, cfg: ChannelConfig) -> pd.DataFrame:
    result = group.copy()
    highs = pd.to_numeric(result["high"], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(result["low"], errors="coerce").astype(float).to_numpy()
    closes = pd.to_numeric(result["close"], errors="coerce").astype(float).to_numpy()
    true_range = highs - lows

    slopes = np.full(len(result), np.nan)
    mids = np.full(len(result), np.nan)
    uppers = np.full(len(result), np.nan)
    lowers = np.full(len(result), np.nan)
    widths = np.full(len(result), np.nan)

    if len(result) >= cfg.lookback:
        x_values = np.arange(cfg.lookback, dtype=float)
        x_mean = float(x_values.mean())
        x_centered = x_values - x_mean
        denominator = float(np.dot(x_centered, x_centered))
        close_windows = np.lib.stride_tricks.sliding_window_view(closes, cfg.lookback)
        finite = np.isfinite(close_windows).all(axis=1)
        if finite.any():
            high_windows = np.lib.stride_tricks.sliding_window_view(highs, cfg.lookback)[finite]
            low_windows = np.lib.stride_tricks.sliding_window_view(lows, cfg.lookback)[finite]
            range_windows = np.lib.stride_tricks.sliding_window_view(true_range, cfg.lookback)[finite]
            valid_closes = close_windows[finite]
            close_means = valid_closes.mean(axis=1)
            valid_slopes = ((valid_closes - close_means[:, None]) @ x_centered) / denominator
            intercepts = close_means - valid_slopes * x_mean
            fitted = intercepts[:, None] + valid_slopes[:, None] * x_values
            valid_mids = fitted[:, -1]
            residual_bands = np.maximum.reduce(
                [
                    np.nanmax(high_windows - fitted, axis=1),
                    np.nanmax(fitted - low_windows, axis=1),
                    np.zeros(len(valid_closes), dtype=float),
                ]
            )
            atr_bands = np.nanmean(range_windows, axis=1) * cfg.band_atr_multiple
            bands = np.maximum(residual_bands, atr_bands)
            target_indexes = cfg.lookback - 1 + np.flatnonzero(finite)
            slopes[target_indexes] = valid_slopes
            mids[target_indexes] = valid_mids
            uppers[target_indexes] = valid_mids + bands
            lowers[target_indexes] = valid_mids - bands
            widths[target_indexes] = bands * 2.0

    result["channel_slope"] = slopes
    result["channel_mid"] = mids
    result["channel_upper"] = uppers
    result["channel_lower"] = lowers
    result["channel_width"] = widths
    result["channel_direction"] = np.select(
        [result["channel_slope"] > cfg.min_slope, result["channel_slope"] < -cfg.min_slope],
        ["up", "down"],
        default="flat",
    )
    return result
