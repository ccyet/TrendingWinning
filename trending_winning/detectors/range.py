from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS, empty_events
from trending_winning.detectors.features import attach_bar_features, rolling_slope_z


@dataclass(frozen=True)
class RangeDetectorConfig:
    """区间识别参数；中部位置只标记禁做，边缘失败突破才给方向事件。"""

    lookback: int = 20
    middle_low: float = 0.25
    middle_high: float = 0.75
    false_break_buffer: float = 0.0
    strong_close_pos: float = 0.65
    min_range_score: float = 0.8
    tick_size: float = 0.01


class RangeDetector:
    """区间模式识别器；独立识别区间中部和失败突破。"""

    name = "range"

    def __init__(self, config: RangeDetectorConfig | None = None) -> None:
        self.config = config or RangeDetectorConfig()
        if self.config.lookback < 3:
            raise ValueError("lookback 至少需要 3。")
        if not 0 <= self.config.middle_low < self.config.middle_high <= 1:
            raise ValueError("middle_low/middle_high 必须在 0 到 1 之间且 low < high。")
        if self.config.min_range_score < 0:
            raise ValueError("min_range_score 不能为负数。")

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        featured = attach_bar_features(bars)
        if featured.empty:
            return empty_events()

        frames: list[pd.DataFrame] = []
        for symbol, group in featured.groupby("stock_code", sort=False):
            scored = self._score_group(group.reset_index(drop=True))
            frame = self._events_for_group(str(symbol), timeframe, scored)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_events()
        return pd.concat(frames, ignore_index=True).loc[:, DETECTOR_EVENT_COLUMNS]

    def _score_group(self, group: pd.DataFrame) -> pd.DataFrame:
        result = group.copy()
        cfg = self.config
        result["range_high"] = result["high"].rolling(cfg.lookback, min_periods=3).max().shift(1)
        result["range_low"] = result["low"].rolling(cfg.lookback, min_periods=3).min().shift(1)
        width = result["range_high"] - result["range_low"]
        result["range_pos"] = (result["close"] - result["range_low"]) / width.replace(0, np.nan)
        previous_low = result["range_low"]
        previous_high = result["range_high"]
        result["failed_breakdown"] = (
            (result["low"] < previous_low * (1.0 - cfg.false_break_buffer))
            & (result["close"] > previous_low)
            & (result["close_pos"] >= cfg.strong_close_pos)
        ).fillna(False)
        result["failed_breakout"] = (
            (result["high"] > previous_high * (1.0 + cfg.false_break_buffer))
            & (result["close"] < previous_high)
            & (result["close_pos"] <= 1.0 - cfg.strong_close_pos)
        ).fillna(False)
        result = self._attach_range_score(result)
        result["is_range_regime"] = result["range_score"] >= cfg.min_range_score
        result["failed_breakdown"] = result["failed_breakdown"] & result["is_range_regime"]
        result["failed_breakout"] = result["failed_breakout"] & result["is_range_regime"]
        result["no_trade_middle"] = (
            result["range_pos"].between(cfg.middle_low, cfg.middle_high, inclusive="both") & result["is_range_regime"]
        )
        return result

    def _attach_range_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        result = frame.copy()
        high = result["high"].astype(float)
        low = result["low"].astype(float)
        close = result["close"].astype(float)
        bar_range = result["bar_range"].astype(float).replace(0, np.nan)
        prev_range = bar_range.shift(1)
        overlap = (pd.concat([high, high.shift(1)], axis=1).min(axis=1) - pd.concat([low, low.shift(1)], axis=1).max(axis=1)).clip(
            lower=0.0
        )
        overlap_base = pd.concat([bar_range, prev_range], axis=1).min(axis=1).replace(0, np.nan)
        result["overlap_ratio"] = (overlap / overlap_base).fillna(0.0).clip(0.0, 1.0)
        result["overlap_mean"] = result["overlap_ratio"].rolling(cfg.lookback, min_periods=cfg.lookback).mean().fillna(0.0)
        result["tail_mean"] = result["tail_ratio"].rolling(cfg.lookback, min_periods=cfg.lookback).mean().fillna(0.0)
        atr = result["atr"].replace(0, np.nan)
        ema_spread = (result["ema_fast"] - result["ema_slow"]).abs()
        result["ema_flatness"] = (1.0 / (1.0 + ema_spread / atr)).fillna(0.0).clip(0.0, 1.0)
        path = close.diff().abs().rolling(cfg.lookback, min_periods=cfg.lookback).sum()
        net_move = (close - close.shift(cfg.lookback - 1)).abs()
        result["directional_efficiency"] = (net_move / path.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
        result["slope_z"] = rolling_slope_z(close, cfg.lookback).fillna(0.0)
        result["failed_break_count"] = (
            (result["failed_breakdown"].astype(int) + result["failed_breakout"].astype(int))
            .rolling(cfg.lookback, min_periods=1)
            .sum()
        )
        score = (
            1.2 * result["overlap_mean"]
            + 0.5 * result["tail_mean"]
            + 0.8 * result["ema_flatness"]
            + 0.25 * result["failed_break_count"]
            - 0.9 * result["directional_efficiency"]
            - 0.6 * result["slope_z"].abs()
        ).clip(lower=0.0)
        result["range_score"] = score.where(path.notna() & net_move.notna(), 0.0)
        return result

    def _events_for_group(self, symbol: str, timeframe: str, scored: pd.DataFrame) -> pd.DataFrame:
        """批量输出区间事件；优先级保持失败下破、失败上破、中部观察。"""
        cfg = self.config
        valid = scored["range_pos"].notna()
        failed_breakdown = valid & scored["failed_breakdown"].fillna(False).astype(bool)
        failed_breakout = valid & ~failed_breakdown & scored["failed_breakout"].fillna(False).astype(bool)
        no_trade_middle = (
            valid
            & ~failed_breakdown
            & ~failed_breakout
            & scored["no_trade_middle"].fillna(False).astype(bool)
        )
        event_mask = failed_breakdown | failed_breakout | no_trade_middle
        if not bool(event_mask.any()):
            return empty_events()

        high = pd.to_numeric(scored["high"], errors="coerce")
        low = pd.to_numeric(scored["low"], errors="coerce")
        close = pd.to_numeric(scored["close"], errors="coerce")
        event_type = pd.Series("", index=scored.index, dtype=object)
        direction = pd.Series("", index=scored.index, dtype=object)
        entry_price = pd.Series(np.nan, index=scored.index, dtype=float)
        stop_price = pd.Series(np.nan, index=scored.index, dtype=float)
        signal_price = pd.Series(np.nan, index=scored.index, dtype=float)

        event_type.loc[failed_breakdown] = "failed_breakdown"
        direction.loc[failed_breakdown] = "long"
        entry_price.loc[failed_breakdown] = high.loc[failed_breakdown] + cfg.tick_size
        stop_price.loc[failed_breakdown] = low.loc[failed_breakdown] - cfg.tick_size
        signal_price.loc[failed_breakdown] = high.loc[failed_breakdown]

        event_type.loc[failed_breakout] = "failed_breakout"
        direction.loc[failed_breakout] = "short"
        entry_price.loc[failed_breakout] = low.loc[failed_breakout] - cfg.tick_size
        stop_price.loc[failed_breakout] = high.loc[failed_breakout] + cfg.tick_size
        signal_price.loc[failed_breakout] = low.loc[failed_breakout]

        event_type.loc[no_trade_middle] = "no_trade_middle"
        direction.loc[no_trade_middle] = "neutral"
        signal_price.loc[no_trade_middle] = close.loc[no_trade_middle]

        selected = scored.loc[event_mask].copy()
        selected_type = event_type.loc[event_mask]
        dates = pd.to_datetime(selected["date"], errors="coerce")
        return pd.DataFrame(
            {
                "event_id": [
                    f"{self.name}:{symbol}:{pd.Timestamp(date).isoformat()}:{event}"
                    for date, event in zip(dates, selected_type, strict=True)
                ],
                "detector_name": self.name,
                "stock_code": symbol,
                "timeframe": timeframe,
                "date": dates.to_numpy(),
                "bar_index": selected.index.astype(int).to_numpy(),
                "event_type": selected_type.to_numpy(),
                "direction": direction.loc[event_mask].to_numpy(),
                "signal_price": signal_price.loc[event_mask].to_numpy(dtype=float),
                "entry_price": entry_price.loc[event_mask].to_numpy(dtype=float),
                "stop_price": stop_price.loc[event_mask].to_numpy(dtype=float),
                "confidence": np.ones(int(event_mask.sum()), dtype=float),
                "metadata": self._metadata_records(selected),
            },
            columns=DETECTOR_EVENT_COLUMNS,
        )

    @staticmethod
    def _metadata_records(scored: pd.DataFrame) -> list[dict[str, object]]:
        return [
            {
                "range_high": float(range_high),
                "range_low": float(range_low),
                "range_pos": float(range_pos),
                "range_score": float(range_score),
                "overlap_mean": float(overlap_mean),
                "tail_mean": float(tail_mean),
                "ema_flatness": float(ema_flatness),
                "directional_efficiency": float(directional_efficiency),
                "failed_break_count": float(failed_break_count),
            }
            for (
                range_high,
                range_low,
                range_pos,
                range_score,
                overlap_mean,
                tail_mean,
                ema_flatness,
                directional_efficiency,
                failed_break_count,
            ) in zip(
                pd.to_numeric(scored["range_high"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["range_low"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["range_pos"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["range_score"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["overlap_mean"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["tail_mean"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["ema_flatness"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["directional_efficiency"], errors="coerce").fillna(0.0),
                pd.to_numeric(scored["failed_break_count"], errors="coerce").fillna(0.0),
                strict=True,
            )
        ]
