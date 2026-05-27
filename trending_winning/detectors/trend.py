from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.detectors.base import DetectorEvent, empty_events, events_to_frame
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

        events: list[DetectorEvent] = []
        for symbol, scored in scored_all.groupby("stock_code", sort=False):
            scored = scored.reset_index(drop=True)
            for index, row in enumerate(scored.to_records(index=False)):
                direction = self._signal_direction(row)
                if direction == "neutral":
                    continue
                entry_price, stop_price, signal_price = self._prices(scored, index, direction)
                if not np.isfinite(entry_price) or not np.isfinite(stop_price) or entry_price <= 0 or stop_price <= 0:
                    continue
                event_type = self._event_type(row, direction)
                event_id = f"{self.name}:{symbol}:{pd.Timestamp(row['date']).isoformat()}:{event_type}"
                events.append(
                    DetectorEvent(
                        event_id=event_id,
                        detector_name=self.name,
                        stock_code=str(symbol),
                        timeframe=timeframe,
                        date=pd.Timestamp(row["date"]),
                        bar_index=int(index),
                        event_type=event_type,
                        direction=direction,
                        signal_price=float(signal_price),
                        entry_price=float(entry_price),
                        stop_price=float(stop_price),
                        confidence=float(min(abs(row["trend_score"]) / max(self.config.min_trend_score, 1e-9), 3.0) / 3.0),
                        metadata={
                            "trend_score": float(row["trend_score"]),
                            "trend_state": str(row["trend_state"]),
                            "pullback_legs": int(row["pullback_legs"]),
                            "slope_z": float(row["slope_z"]) if pd.notna(row["slope_z"]) else 0.0,
                            "close_pos": float(row["close_pos"]),
                            "body_ratio": float(row["body_ratio"]),
                        },
                    )
                )
        return events_to_frame(events)

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

    def _signal_direction(self, row: object) -> str:
        cfg = self.config
        if (
            float(row["trend_score"]) >= cfg.min_trend_score
            and str(row["trend_state"]) == "bull"
            and float(row["close_pos"]) >= cfg.strong_close_pos
            and float(row["body_ratio"]) >= cfg.min_body_ratio
            and float(row["close"]) > float(row["open"])
        ):
            return "long"
        if (
            float(row["trend_score"]) <= -cfg.min_trend_score
            and str(row["trend_state"]) == "bear"
            and float(row["close_pos"]) <= 1.0 - cfg.strong_close_pos
            and float(row["body_ratio"]) >= cfg.min_body_ratio
            and float(row["close"]) < float(row["open"])
        ):
            return "short"
        return "neutral"

    def _event_type(self, row: object, direction: str) -> str:
        legs = int(row["pullback_legs"])
        if direction == "long":
            return "bull_h2_setup" if legs >= self.config.h2_min_pullback_legs else "bull_h1_setup"
        return "bear_l2_setup" if legs >= self.config.h2_min_pullback_legs else "bear_l1_setup"

    def _prices(self, group: pd.DataFrame, index: int, direction: str) -> tuple[float, float, float]:
        cfg = self.config
        row = group.loc[index]
        start = max(0, index - cfg.pullback_lookback + 1)
        if direction == "long":
            stop_base = float(group.loc[start:index, "low"].min())
            return float(row["high"] + cfg.tick_size), stop_base - cfg.tick_size, float(row["high"])
        stop_base = float(group.loc[start:index, "high"].max())
        return float(row["low"] - cfg.tick_size), stop_base + cfg.tick_size, float(row["low"])


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
