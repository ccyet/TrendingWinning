from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.data.schema import normalize_bars
from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS, empty_events
from trending_winning.detectors.pivots import confirmed_pivots


@dataclass(frozen=True)
class ChannelDetectorConfig:
    """趋势通道参数；支持滚动 log 回归和摆动点人工画线两种算法。"""

    lookback: int = 40
    sigma_multiple: float = 2.0
    break_buffer: float = 0.0
    tick_size: float = 0.01
    channel_method: str = "regression"
    swing_left_bars: int = 2
    swing_right_bars: int = 2


class ChannelDetector:
    """通道突破识别器；只输出通道事件，不判断趋势或反转。"""

    name = "channel"

    def __init__(self, config: ChannelDetectorConfig | None = None) -> None:
        self.config = config or ChannelDetectorConfig()
        if self.config.lookback < 3:
            raise ValueError("lookback 至少需要 3。")
        if self.config.sigma_multiple <= 0:
            raise ValueError("sigma_multiple 必须大于 0。")
        if self.config.break_buffer < 0:
            raise ValueError("break_buffer 不能为负数。")
        if self.config.tick_size <= 0:
            raise ValueError("tick_size 必须大于 0。")
        if self.config.channel_method not in {"regression", "swing"}:
            raise ValueError("channel_method 仅支持 regression 或 swing。")
        if self.config.swing_left_bars < 1 or self.config.swing_right_bars < 1:
            raise ValueError("swing_left_bars 和 swing_right_bars 至少需要 1。")

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        channeled = (
            attach_swing_trend_channel(bars, self.config)
            if self.config.channel_method == "swing"
            else attach_log_regression_channel(bars, self.config)
        )
        if channeled.empty:
            return empty_events()

        frames: list[pd.DataFrame] = []
        for _, group in channeled.groupby("stock_code", sort=False):
            group = group.reset_index(drop=True)
            frame = self._events_for_group(group, timeframe=timeframe)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_events()
        return pd.concat(frames, ignore_index=True).loc[:, DETECTOR_EVENT_COLUMNS]

    def _events_for_group(self, group: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
        """批量生成通道突破事件；通道自身只负责边界突破，不混入趋势判断。"""
        valid_channel = group["channel_upper"].notna() & group["channel_lower"].notna()
        upper = pd.to_numeric(group["prev_channel_upper"], errors="coerce")
        lower = pd.to_numeric(group["prev_channel_lower"], errors="coerce")
        close = pd.to_numeric(group["close"], errors="coerce")
        high = pd.to_numeric(group["high"], errors="coerce")
        low = pd.to_numeric(group["low"], errors="coerce")
        valid = valid_channel & upper.notna() & lower.notna()
        overshoot_up = valid & close.gt(upper * (1.0 + self.config.break_buffer))
        break_down = valid & ~overshoot_up & close.lt(lower * (1.0 - self.config.break_buffer))
        event_mask = overshoot_up | break_down
        if not bool(event_mask.any()):
            return empty_events()

        event_type = pd.Series("", index=group.index, dtype=object)
        direction = pd.Series("", index=group.index, dtype=object)
        signal_price = pd.Series(np.nan, index=group.index, dtype=float)
        entry_price = pd.Series(np.nan, index=group.index, dtype=float)
        stop_price = pd.Series(np.nan, index=group.index, dtype=float)

        event_type.loc[overshoot_up] = "channel_overshoot_up"
        direction.loc[overshoot_up] = "long"
        signal_price.loc[overshoot_up] = high.loc[overshoot_up]
        entry_price.loc[overshoot_up] = high.loc[overshoot_up] + self.config.tick_size
        stop_price.loc[overshoot_up] = lower.loc[overshoot_up]

        event_type.loc[break_down] = "channel_break_down"
        direction.loc[break_down] = "short"
        signal_price.loc[break_down] = low.loc[break_down]
        entry_price.loc[break_down] = low.loc[break_down] - self.config.tick_size
        stop_price.loc[break_down] = upper.loc[break_down]

        selected = group.loc[event_mask].copy()
        selected_type = event_type.loc[event_mask]
        selected_symbol = selected["stock_code"].astype(str)
        dates = pd.to_datetime(selected["date"], errors="coerce")
        return pd.DataFrame(
            {
                "event_id": [
                    f"{self.name}:{symbol}:{pd.Timestamp(date).isoformat()}:{event}"
                    for symbol, date, event in zip(selected_symbol, dates, selected_type, strict=True)
                ],
                "detector_name": self.name,
                "stock_code": selected_symbol.to_numpy(),
                "timeframe": timeframe,
                "date": dates.to_numpy(),
                "bar_index": selected.index.astype(int).to_numpy(),
                "event_type": selected_type.to_numpy(),
                "direction": direction.loc[event_mask].to_numpy(),
                "signal_price": signal_price.loc[event_mask].to_numpy(dtype=float),
                "entry_price": entry_price.loc[event_mask].to_numpy(dtype=float),
                "stop_price": stop_price.loc[event_mask].to_numpy(dtype=float),
                "confidence": np.ones(int(event_mask.sum()), dtype=float),
                "metadata": self._metadata_records(selected, upper.loc[event_mask], lower.loc[event_mask]),
            },
            columns=DETECTOR_EVENT_COLUMNS,
        )

    def _metadata_records(
        self,
        selected: pd.DataFrame,
        upper: pd.Series,
        lower: pd.Series,
    ) -> list[dict[str, object]]:
        return [
            {
                "channel_upper": float(channel_upper),
                "channel_lower": float(channel_lower),
                "channel_slope": float(channel_slope),
                "channel_pos": float(channel_pos),
                "channel_sigma": float(channel_sigma),
                "channel_r2": _metadata_float(channel_r2),
                "channel_method": str(channel_method),
                "channel_anchor_index_1": _metadata_float(channel_anchor_index_1),
                "channel_anchor_index_2": _metadata_float(channel_anchor_index_2),
            }
            for (
                channel_upper,
                channel_lower,
                channel_slope,
                channel_pos,
                channel_sigma,
                channel_r2,
                channel_method,
                channel_anchor_index_1,
                channel_anchor_index_2,
            ) in zip(
                pd.to_numeric(upper, errors="coerce").fillna(0.0),
                pd.to_numeric(lower, errors="coerce").fillna(0.0),
                pd.to_numeric(selected["channel_slope"], errors="coerce").fillna(0.0),
                pd.to_numeric(selected["channel_pos"], errors="coerce").fillna(0.0),
                pd.to_numeric(selected["channel_sigma"], errors="coerce").fillna(0.0),
                selected.get("channel_r2", pd.Series([pd.NA] * len(selected), index=selected.index)),
                selected.get(
                    "channel_method",
                    pd.Series([self.config.channel_method] * len(selected), index=selected.index),
                ),
                selected.get("channel_anchor_index_1", pd.Series([pd.NA] * len(selected), index=selected.index)),
                selected.get("channel_anchor_index_2", pd.Series([pd.NA] * len(selected), index=selected.index)),
                strict=True,
            )
        ]


def attach_log_regression_channel(bars: pd.DataFrame, config: ChannelDetectorConfig | None = None) -> pd.DataFrame:
    cfg = config or ChannelDetectorConfig()
    normalized = normalize_bars(bars)
    if normalized.empty:
        return normalized.assign(
            channel_mid=pd.Series(dtype=float),
            channel_upper=pd.Series(dtype=float),
            channel_lower=pd.Series(dtype=float),
            prev_channel_upper=pd.Series(dtype=float),
            prev_channel_lower=pd.Series(dtype=float),
            channel_slope=pd.Series(dtype=float),
            channel_sigma=pd.Series(dtype=float),
            channel_r2=pd.Series(dtype=float),
            channel_pos=pd.Series(dtype=float),
            channel_method=pd.Series(dtype=str),
            channel_anchor_index_1=pd.Series(dtype=float),
            channel_anchor_index_2=pd.Series(dtype=float),
        )

    frames: list[pd.DataFrame] = []
    for _, group in normalized.groupby("stock_code", sort=False):
        frames.append(_attach_group_log_channel(group.reset_index(drop=True), cfg))
    return pd.concat(frames, ignore_index=True)


def _attach_group_log_channel(group: pd.DataFrame, cfg: ChannelDetectorConfig) -> pd.DataFrame:
    result = group.copy()
    close = result["close"].astype(float).to_numpy()
    log_close = np.log(np.where(close > 0, close, np.nan))
    mid, upper, lower, slope, sigma, r2 = _rolling_log_regression_channel(log_close, cfg)

    result["channel_mid"] = mid
    result["channel_upper"] = upper
    result["channel_lower"] = lower
    result["prev_channel_upper"] = pd.Series(upper).shift(1).to_numpy()
    result["prev_channel_lower"] = pd.Series(lower).shift(1).to_numpy()
    result["channel_slope"] = slope
    result["channel_sigma"] = sigma
    result["channel_r2"] = r2
    result["channel_pos"] = (result["close"] - result["channel_lower"]) / (result["channel_upper"] - result["channel_lower"])
    result["channel_method"] = "regression"
    result["channel_anchor_index_1"] = np.nan
    result["channel_anchor_index_2"] = np.nan
    return result


def _rolling_log_regression_channel(
    log_close: np.ndarray,
    cfg: ChannelDetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = len(log_close)
    mid = np.full(length, np.nan)
    upper = np.full(length, np.nan)
    lower = np.full(length, np.nan)
    slope = np.full(length, np.nan)
    sigma = np.full(length, np.nan)
    r2 = np.full(length, np.nan)
    if length < cfg.lookback:
        return mid, upper, lower, slope, sigma, r2

    x = np.arange(cfg.lookback, dtype=float)
    x_mean = float(x.mean())
    x_centered = x - x_mean
    denominator = float(np.dot(x_centered, x_centered))
    windows = np.lib.stride_tricks.sliding_window_view(log_close, cfg.lookback)
    finite = np.isfinite(windows).all(axis=1)
    if not finite.any():
        return mid, upper, lower, slope, sigma, r2

    valid_windows = windows[finite]
    y_mean = valid_windows.mean(axis=1)
    beta = ((valid_windows - y_mean[:, None]) @ x_centered) / denominator
    alpha = y_mean - beta * x_mean
    fitted = alpha[:, None] + beta[:, None] * x
    residual = valid_windows - fitted
    residual_median = np.median(residual, axis=1)
    mad = np.median(np.abs(residual - residual_median[:, None]), axis=1)
    std = np.std(residual, axis=1, ddof=1) if cfg.lookback > 1 else np.zeros(len(valid_windows))
    robust_sigma = np.maximum.reduce([1.4826 * mad, std, np.full(len(valid_windows), 1e-9)])
    ss_res = np.sum(residual * residual, axis=1)
    ss_tot = np.sum((valid_windows - y_mean[:, None]) ** 2, axis=1)
    fit_quality = np.where(ss_tot > 1e-18, 1.0 - ss_res / ss_tot, np.where(ss_res <= 1e-18, 1.0, 0.0))
    fit_quality = np.clip(fit_quality, 0.0, 1.0)

    target_index = cfg.lookback - 1 + np.flatnonzero(finite)
    current_mid_log = fitted[:, -1]
    mid[target_index] = np.exp(current_mid_log)
    upper[target_index] = np.exp(current_mid_log + cfg.sigma_multiple * robust_sigma)
    lower[target_index] = np.exp(current_mid_log - cfg.sigma_multiple * robust_sigma)
    slope[target_index] = beta
    sigma[target_index] = robust_sigma
    r2[target_index] = fit_quality
    return mid, upper, lower, slope, sigma, r2


def attach_swing_trend_channel(bars: pd.DataFrame, config: ChannelDetectorConfig | None = None) -> pd.DataFrame:
    cfg = config or ChannelDetectorConfig(channel_method="swing")
    normalized = normalize_bars(bars)
    if normalized.empty:
        return normalized.assign(
            channel_mid=pd.Series(dtype=float),
            channel_upper=pd.Series(dtype=float),
            channel_lower=pd.Series(dtype=float),
            prev_channel_upper=pd.Series(dtype=float),
            prev_channel_lower=pd.Series(dtype=float),
            channel_slope=pd.Series(dtype=float),
            channel_sigma=pd.Series(dtype=float),
            channel_r2=pd.Series(dtype=float),
            channel_pos=pd.Series(dtype=float),
            channel_method=pd.Series(dtype=str),
            channel_anchor_index_1=pd.Series(dtype=float),
            channel_anchor_index_2=pd.Series(dtype=float),
        )

    frames: list[pd.DataFrame] = []
    for _, group in normalized.groupby("stock_code", sort=False):
        frames.append(_attach_group_swing_channel(group.reset_index(drop=True), cfg))
    return pd.concat(frames, ignore_index=True)


def _attach_group_swing_channel(group: pd.DataFrame, cfg: ChannelDetectorConfig) -> pd.DataFrame:
    result = group.copy()
    length = len(result)
    high = result["high"].astype(float).to_numpy()
    low = result["low"].astype(float).to_numpy()
    close = result["close"].astype(float).to_numpy()
    pivot_high, pivot_low = confirmed_pivots(high, low, cfg.swing_left_bars, cfg.swing_right_bars)
    mid = np.full(length, np.nan)
    upper = np.full(length, np.nan)
    lower = np.full(length, np.nan)
    slope = np.full(length, np.nan)
    sigma = np.full(length, np.nan)
    r2 = np.full(length, np.nan)
    anchor_one = np.full(length, np.nan)
    anchor_two = np.full(length, np.nan)
    low_anchors: list[int] = []
    high_anchors: list[int] = []

    for index in range(length):
        confirmed_index = index - cfg.swing_right_bars
        if confirmed_index >= 0:
            if pivot_low[confirmed_index]:
                low_anchors.append(int(confirmed_index))
            if pivot_high[confirmed_index]:
                high_anchors.append(int(confirmed_index))
        if len(low_anchors) >= 2:
            first, second = low_anchors[-2], low_anchors[-1]
            beta = (low[second] - low[first]) / (second - first)
            start = max(0, index - cfg.lookback + 1)
            window_indexes = np.arange(start, index + 1, dtype=float)
            support_window = low[first] + beta * (window_indexes - first)
            support_value = float(low[first] + beta * (index - first))
            width = max(float(np.nanmax(high[start : index + 1] - support_window)), 1e-9)
            lower[index] = support_value
            upper[index] = support_value + width
            mid[index] = (upper[index] + lower[index]) / 2.0
            slope[index] = beta
            sigma[index] = width / 2.0
            anchor_one[index] = first
            anchor_two[index] = second
        elif len(high_anchors) >= 2:
            first, second = high_anchors[-2], high_anchors[-1]
            beta = (high[second] - high[first]) / (second - first)
            start = max(0, index - cfg.lookback + 1)
            window_indexes = np.arange(start, index + 1, dtype=float)
            resistance_window = high[first] + beta * (window_indexes - first)
            resistance_value = float(high[first] + beta * (index - first))
            width = max(float(np.nanmax(resistance_window - low[start : index + 1])), 1e-9)
            upper[index] = resistance_value
            lower[index] = resistance_value - width
            mid[index] = (upper[index] + lower[index]) / 2.0
            slope[index] = beta
            sigma[index] = width / 2.0
            anchor_one[index] = first
            anchor_two[index] = second

    result["channel_mid"] = mid
    result["channel_upper"] = upper
    result["channel_lower"] = lower
    result["prev_channel_upper"] = pd.Series(upper).shift(1).to_numpy()
    result["prev_channel_lower"] = pd.Series(lower).shift(1).to_numpy()
    result["channel_slope"] = slope
    result["channel_sigma"] = sigma
    result["channel_r2"] = r2
    result["channel_pos"] = (close - result["channel_lower"]) / (result["channel_upper"] - result["channel_lower"])
    result["channel_method"] = "swing"
    result["channel_anchor_index_1"] = anchor_one
    result["channel_anchor_index_2"] = anchor_two
    return result


def _metadata_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
