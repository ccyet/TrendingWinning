from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS, empty_events
from trending_winning.detectors.features import attach_bar_features, rolling_slope_z


@dataclass(frozen=True)
class TrendDetectorConfig:
    """趋势识别参数；组合斜率、结构、收盘强度和跟随 K 评分。"""

    lookback: int = 20
    min_trend_score: float = 1.0
    strong_close_pos: float = 0.65
    min_body_ratio: float = 0.45
    pullback_lookback: int = 5
    h2_min_pullback_legs: int = 2
    tick_size: float = 0.01


class TrendDetector:
    """趋势模式识别器；只负责输出趋势方向的信号 K 事件。"""

    name = "trend"

    def __init__(self, config: TrendDetectorConfig | None = None) -> None:
        self.config = config or TrendDetectorConfig()
        if self.config.lookback < 3:
            raise ValueError("lookback 至少需要 3。")
        if not 0 < self.config.strong_close_pos < 1:
            raise ValueError("strong_close_pos 必须在 0 到 1 之间。")
        if not 0 <= self.config.min_body_ratio <= 1:
            raise ValueError("min_body_ratio 必须在 0 到 1 之间。")
        if self.config.pullback_lookback < 1:
            raise ValueError("pullback_lookback 至少需要 1。")
        if self.config.h2_min_pullback_legs < 1:
            raise ValueError("h2_min_pullback_legs 至少需要 1。")

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        scored_all = attach_trend_state(bars, self.config)
        if scored_all.empty:
            return empty_events()

        frames: list[pd.DataFrame] = []
        for symbol, scored in scored_all.groupby("stock_code", sort=False):
            frame = self._events_for_group(str(symbol), timeframe, scored.reset_index(drop=True))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_events()
        return pd.concat(frames, ignore_index=True).loc[:, DETECTOR_EVENT_COLUMNS]

    def _score_group(self, group: pd.DataFrame) -> pd.DataFrame:
        result = group.copy()
        cfg = self.config
        close = result["close"].astype(float)
        high = result["high"].astype(float)
        low = result["low"].astype(float)
        result["slope_z"] = rolling_slope_z(close, cfg.lookback).fillna(0.0)
        higher_high = high > high.shift(1)
        higher_low = low > low.shift(1)
        lower_high = high < high.shift(1)
        lower_low = low < low.shift(1)
        structure = (
            higher_high.rolling(cfg.lookback, min_periods=2).sum()
            + higher_low.rolling(cfg.lookback, min_periods=2).sum()
            - lower_high.rolling(cfg.lookback, min_periods=2).sum()
            - lower_low.rolling(cfg.lookback, min_periods=2).sum()
        ) / cfg.lookback
        close_quality = (2.0 * result["close_pos"] - 1.0).rolling(cfg.lookback, min_periods=2).mean()
        follow = (
            result["follow_up"].rolling(cfg.lookback, min_periods=2).sum()
            - result["follow_down"].rolling(cfg.lookback, min_periods=2).sum()
        ) / cfg.lookback
        result["trend_score"] = (
            result["slope_z"].fillna(0.0)
            + structure.fillna(0.0)
            + result["ma_align"].fillna(0.0) * 0.5
            + close_quality.fillna(0.0)
            + follow.fillna(0.0)
        )
        result["trend_state"] = np.select(
            [result["trend_score"] >= cfg.min_trend_score, result["trend_score"] <= -cfg.min_trend_score],
            ["bull", "bear"],
            default="neutral",
        )
        result["bull_pullback_legs"] = _count_pullback_legs(close, direction="bull", lookback=cfg.pullback_lookback)
        result["bear_pullback_legs"] = _count_pullback_legs(close, direction="bear", lookback=cfg.pullback_lookback)
        result["pullback_legs"] = np.select(
            [result["trend_state"] == "bull", result["trend_state"] == "bear"],
            [result["bull_pullback_legs"], result["bear_pullback_legs"]],
            default=0,
        )
        return result

    def _events_for_group(self, symbol: str, timeframe: str, scored: pd.DataFrame) -> pd.DataFrame:
        """批量生成趋势事件；滚动计算后只按事件行组装 metadata。"""
        direction = self._signal_direction_series(scored)
        entry_price, stop_price, signal_price = self._price_series(scored, direction)
        price_mask = entry_price.gt(0) & stop_price.gt(0) & np.isfinite(entry_price) & np.isfinite(stop_price)
        event_mask = direction.ne("neutral") & price_mask
        if not bool(event_mask.any()):
            return empty_events()

        selected = scored.loc[event_mask].copy()
        selected_direction = direction.loc[event_mask]
        selected_entry = entry_price.loc[event_mask].astype(float)
        selected_stop = stop_price.loc[event_mask].astype(float)
        selected_signal = signal_price.loc[event_mask].astype(float)
        event_type = self._event_type_series(selected, selected_direction)
        confidence = self._confidence_series(selected)
        metadata = self._metadata_records(selected)
        dates = pd.to_datetime(selected["date"], errors="coerce")

        return pd.DataFrame(
            {
                "event_id": [
                    f"{self.name}:{symbol}:{pd.Timestamp(date).isoformat()}:{event}"
                    for date, event in zip(dates, event_type, strict=True)
                ],
                "detector_name": self.name,
                "stock_code": symbol,
                "timeframe": timeframe,
                "date": dates.to_numpy(),
                "bar_index": selected.index.astype(int).to_numpy(),
                "event_type": event_type.to_numpy(),
                "direction": selected_direction.to_numpy(),
                "signal_price": selected_signal.to_numpy(),
                "entry_price": selected_entry.to_numpy(),
                "stop_price": selected_stop.to_numpy(),
                "confidence": confidence.to_numpy(dtype=float),
                "metadata": metadata,
            },
            columns=DETECTOR_EVENT_COLUMNS,
        )

    def _signal_direction_series(self, scored: pd.DataFrame) -> pd.Series:
        cfg = self.config
        trend_score = pd.to_numeric(scored["trend_score"], errors="coerce")
        close_pos = pd.to_numeric(scored["close_pos"], errors="coerce")
        body_ratio = pd.to_numeric(scored["body_ratio"], errors="coerce")
        close = pd.to_numeric(scored["close"], errors="coerce")
        open_ = pd.to_numeric(scored["open"], errors="coerce")
        trend_state = scored["trend_state"].astype(str)
        long_mask = (
            trend_score.ge(cfg.min_trend_score)
            & trend_state.eq("bull")
            & close_pos.ge(cfg.strong_close_pos)
            & body_ratio.ge(cfg.min_body_ratio)
            & close.gt(open_)
        )
        short_mask = (
            trend_score.le(-cfg.min_trend_score)
            & trend_state.eq("bear")
            & close_pos.le(1.0 - cfg.strong_close_pos)
            & body_ratio.ge(cfg.min_body_ratio)
            & close.lt(open_)
        )
        return pd.Series(np.select([long_mask, short_mask], ["long", "short"], default="neutral"), index=scored.index)

    def _event_type_series(self, scored: pd.DataFrame, direction: pd.Series) -> pd.Series:
        legs = pd.to_numeric(scored["pullback_legs"], errors="coerce").fillna(0).astype(int)
        h2 = legs.ge(self.config.h2_min_pullback_legs)
        event_type = np.select(
            [
                direction.eq("long") & h2,
                direction.eq("long") & ~h2,
                direction.eq("short") & h2,
                direction.eq("short") & ~h2,
            ],
            ["bull_h2_setup", "bull_h1_setup", "bear_l2_setup", "bear_l1_setup"],
            default="",
        )
        return pd.Series(event_type, index=scored.index)

    def _price_series(self, group: pd.DataFrame, direction: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        cfg = self.config
        high = pd.to_numeric(group["high"], errors="coerce")
        low = pd.to_numeric(group["low"], errors="coerce")
        rolling_low = low.rolling(cfg.pullback_lookback, min_periods=1).min()
        rolling_high = high.rolling(cfg.pullback_lookback, min_periods=1).max()
        long_mask = direction.eq("long")
        short_mask = direction.eq("short")
        entry_price = pd.Series(np.nan, index=group.index, dtype=float)
        stop_price = pd.Series(np.nan, index=group.index, dtype=float)
        signal_price = pd.Series(np.nan, index=group.index, dtype=float)
        entry_price.loc[long_mask] = high.loc[long_mask] + cfg.tick_size
        stop_price.loc[long_mask] = rolling_low.loc[long_mask] - cfg.tick_size
        signal_price.loc[long_mask] = high.loc[long_mask]
        entry_price.loc[short_mask] = low.loc[short_mask] - cfg.tick_size
        stop_price.loc[short_mask] = rolling_high.loc[short_mask] + cfg.tick_size
        signal_price.loc[short_mask] = low.loc[short_mask]
        return entry_price, stop_price, signal_price

    def _confidence_series(self, scored: pd.DataFrame) -> pd.Series:
        denominator = max(self.config.min_trend_score, 1e-9)
        score = pd.to_numeric(scored["trend_score"], errors="coerce").abs().fillna(0.0)
        return (score / denominator).clip(upper=3.0) / 3.0

    @staticmethod
    def _metadata_records(scored: pd.DataFrame) -> list[dict[str, object]]:
        slope_z = pd.to_numeric(scored["slope_z"], errors="coerce")
        return [
            {
                "trend_score": float(trend_score),
                "trend_state": str(trend_state),
                "pullback_legs": int(pullback_legs),
                "slope_z": float(slope) if pd.notna(slope) else 0.0,
                "close_pos": float(close_pos),
                "body_ratio": float(body_ratio),
            }
            for trend_score, trend_state, pullback_legs, slope, close_pos, body_ratio in zip(
                pd.to_numeric(scored["trend_score"], errors="coerce").fillna(0.0),
                scored["trend_state"],
                pd.to_numeric(scored["pullback_legs"], errors="coerce").fillna(0).astype(int),
                slope_z,
                pd.to_numeric(scored["close_pos"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["body_ratio"], errors="coerce").fillna(0.0),
                strict=True,
            )
        ]


def _count_pullback_legs(close: pd.Series, *, direction: str, lookback: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    if len(values) == 0:
        return pd.Series(np.array([], dtype=int), index=close.index)

    deltas = np.diff(values, prepend=np.nan)
    pullback = deltas < 0 if direction == "bull" else deltas > 0
    leg_starts = pullback & ~np.r_[False, pullback[:-1]]
    prefix_starts = np.r_[0, np.cumsum(leg_starts.astype(int))]

    indexes = np.arange(len(values))
    starts = np.maximum(1, indexes - lookback)
    has_window = starts < indexes
    safe_starts = np.minimum(starts, len(values) - 1)
    safe_after_starts = np.minimum(starts + 1, len(prefix_starts) - 1)

    boundary_legs = (has_window & pullback[safe_starts]).astype(int)
    inner_legs = prefix_starts[indexes] - prefix_starts[safe_after_starts]
    out = np.where(has_window, inner_legs + boundary_legs, 0).astype(int)
    return pd.Series(out, index=close.index)


def attach_trend_state(bars: pd.DataFrame, config: TrendDetectorConfig | None = None) -> pd.DataFrame:
    """给每根 K 线打趋势状态；多周期门控可复用，不需要重新跑订单策略。"""
    cfg = config or TrendDetectorConfig()
    featured = attach_bar_features(bars)
    if featured.empty:
        return featured.assign(
            slope_z=pd.Series(dtype=float),
            trend_score=pd.Series(dtype=float),
            trend_state=pd.Series(dtype=str),
            bull_pullback_legs=pd.Series(dtype=int),
            bear_pullback_legs=pd.Series(dtype=int),
            pullback_legs=pd.Series(dtype=int),
        )
    detector = TrendDetector(cfg)
    frames = [
        detector._score_group(group.reset_index(drop=True))
        for _, group in featured.groupby("stock_code", sort=False)
    ]
    return pd.concat(frames, ignore_index=True)
