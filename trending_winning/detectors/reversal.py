from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS, empty_events
from trending_winning.detectors.features import attach_bar_features, rolling_slope_z


@dataclass(frozen=True)
class ReversalDetectorConfig:
    """反转识别参数；默认把第一次反转作为观察，第二次反转才可交易。"""

    lookback: int = 20
    strong_close_pos: float = 0.65
    min_body_ratio: float = 0.45
    old_extreme_tolerance_pct: float = 0.01
    require_old_extreme_test: bool = True
    require_structure_confirmation: bool = True
    tick_size: float = 0.01


class ReversalDetector:
    """反转模式识别器；和趋势、通道、区间检测完全分离。"""

    name = "reversal"

    def __init__(self, config: ReversalDetectorConfig | None = None) -> None:
        self.config = config or ReversalDetectorConfig()
        if self.config.lookback < 3:
            raise ValueError("lookback 至少需要 3。")
        if not 0 < self.config.strong_close_pos < 1:
            raise ValueError("strong_close_pos 必须在 0 到 1 之间。")
        if not 0 <= self.config.min_body_ratio <= 1:
            raise ValueError("min_body_ratio 必须在 0 到 1 之间。")
        if self.config.old_extreme_tolerance_pct < 0:
            raise ValueError("old_extreme_tolerance_pct 不能为负数。")

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        featured = attach_bar_features(bars)
        if featured.empty:
            return empty_events()

        frames: list[pd.DataFrame] = []
        for symbol, group in featured.groupby("stock_code", sort=False):
            scored = group.reset_index(drop=True).copy()
            scored["slope_z"] = rolling_slope_z(scored["close"], self.config.lookback).fillna(0.0)
            scored["_prior_slope"] = scored["slope_z"].shift(1)
            frame = self._events_for_group(str(symbol), timeframe, scored)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return empty_events()
        return pd.concat(frames, ignore_index=True).loc[:, DETECTOR_EVENT_COLUMNS]

    def _events_for_group(self, symbol: str, timeframe: str, scored: pd.DataFrame) -> pd.DataFrame:
        """只遍历可能成为反转的候选 K；watch 状态保持显式，避免全量逐 K 记录扫描。"""
        first_bull = self._first_bull_reversal_mask(scored)
        first_bear = self._first_bear_reversal_mask(scored)
        bull_follow = self._bull_follow_through_mask(scored)
        bear_follow = self._bear_follow_through_mask(scored)
        candidate_mask = first_bull | first_bear | bull_follow | bear_follow
        if not bool(candidate_mask.any()):
            return empty_events()

        event_rows: list[dict[str, object]] = []
        bull_watch: dict[str, object] | None = None
        bear_watch: dict[str, object] | None = None
        for index in np.flatnonzero(candidate_mask.to_numpy(dtype=bool)):
            row = scored.iloc[int(index)]
            event_type = ""
            direction = "neutral"
            metadata: dict[str, object] | None = None
            if bool(first_bull.iloc[int(index)]):
                event_type = "first_reversal_watch_long"
                bull_watch = self._new_watch(scored, int(index), "long")
                metadata = self._metadata(row, bull_watch, old_extreme_failed=False, structure_confirmed=False)
            elif bool(first_bear.iloc[int(index)]):
                event_type = "first_reversal_watch_short"
                bear_watch = self._new_watch(scored, int(index), "short")
                metadata = self._metadata(row, bear_watch, old_extreme_failed=False, structure_confirmed=False)
            elif self._can_trade_second_reversal(bull_watch, int(index)) and bool(bull_follow.iloc[int(index)]):
                old_extreme_failed = self._old_extreme_test_failed(scored, bull_watch, int(index))
                structure_confirmed = self._structure_confirmed(row, bull_watch)
                if not self._second_reversal_confirmed(old_extreme_failed, structure_confirmed):
                    continue
                event_type = "second_reversal_long"
                direction = "long"
                bull_watch["used"] = True
                metadata = self._metadata(row, bull_watch, old_extreme_failed, structure_confirmed)
            elif self._can_trade_second_reversal(bear_watch, int(index)) and bool(bear_follow.iloc[int(index)]):
                old_extreme_failed = self._old_extreme_test_failed(scored, bear_watch, int(index))
                structure_confirmed = self._structure_confirmed(row, bear_watch)
                if not self._second_reversal_confirmed(old_extreme_failed, structure_confirmed):
                    continue
                event_type = "second_reversal_short"
                direction = "short"
                bear_watch["used"] = True
                metadata = self._metadata(row, bear_watch, old_extreme_failed, structure_confirmed)
            if not event_type or metadata is None:
                continue
            event_rows.append(
                self._event_record(
                    symbol=symbol,
                    timeframe=timeframe,
                    row=row,
                    index=int(index),
                    event_type=event_type,
                    direction=direction,
                    metadata=metadata,
                )
            )
        if not event_rows:
            return empty_events()
        return pd.DataFrame(event_rows, columns=DETECTOR_EVENT_COLUMNS)

    def _event_record(
        self,
        *,
        symbol: str,
        timeframe: str,
        row: pd.Series,
        index: int,
        event_type: str,
        direction: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        entry, stop, signal = self._event_prices(row, direction, event_type)
        return {
            "event_id": f"{self.name}:{symbol}:{pd.Timestamp(row['date']).isoformat()}:{event_type}",
            "detector_name": self.name,
            "stock_code": symbol,
            "timeframe": timeframe,
            "date": pd.Timestamp(row["date"]),
            "bar_index": int(index),
            "event_type": event_type,
            "direction": direction,
            "signal_price": signal,
            "entry_price": entry,
            "stop_price": stop,
            "confidence": 0.5 if direction == "neutral" else 1.0,
            "metadata": metadata,
        }

    def _event_prices(self, row: object, direction: str, event_type: str) -> tuple[float, float, float]:
        if direction == "long":
            return (
                float(row["high"] + self.config.tick_size),
                float(row["low"] - self.config.tick_size),
                float(row["high"]),
            )
        if direction == "short":
            return (
                float(row["low"] - self.config.tick_size),
                float(row["high"] + self.config.tick_size),
                float(row["low"]),
            )
        if event_type.endswith("_short"):
            return float(row["close"]), float(row["high"] + self.config.tick_size), float(row["low"])
        return float(row["close"]), float(row["low"] - self.config.tick_size), float(row["high"])

    def _first_bull_reversal(self, row: object) -> bool:
        return (
            float(row["_prior_slope"]) < 0
            and float(row["slope_z"]) >= 0
            and float(row["close_pos"]) >= self.config.strong_close_pos
            and float(row["body_ratio"]) >= self.config.min_body_ratio
            and float(row["close"]) > float(row["open"])
        )

    def _first_bull_reversal_mask(self, scored: pd.DataFrame) -> pd.Series:
        return (
            pd.to_numeric(scored["_prior_slope"], errors="coerce").lt(0)
            & pd.to_numeric(scored["slope_z"], errors="coerce").ge(0)
            & pd.to_numeric(scored["close_pos"], errors="coerce").ge(self.config.strong_close_pos)
            & pd.to_numeric(scored["body_ratio"], errors="coerce").ge(self.config.min_body_ratio)
            & pd.to_numeric(scored["close"], errors="coerce").gt(pd.to_numeric(scored["open"], errors="coerce"))
        )

    def _first_bear_reversal(self, row: object) -> bool:
        return (
            float(row["_prior_slope"]) > 0
            and float(row["slope_z"]) <= 0
            and float(row["close_pos"]) <= 1.0 - self.config.strong_close_pos
            and float(row["body_ratio"]) >= self.config.min_body_ratio
            and float(row["close"]) < float(row["open"])
        )

    def _first_bear_reversal_mask(self, scored: pd.DataFrame) -> pd.Series:
        return (
            pd.to_numeric(scored["_prior_slope"], errors="coerce").gt(0)
            & pd.to_numeric(scored["slope_z"], errors="coerce").le(0)
            & pd.to_numeric(scored["close_pos"], errors="coerce").le(1.0 - self.config.strong_close_pos)
            & pd.to_numeric(scored["body_ratio"], errors="coerce").ge(self.config.min_body_ratio)
            & pd.to_numeric(scored["close"], errors="coerce").lt(pd.to_numeric(scored["open"], errors="coerce"))
        )

    def _new_watch(self, scored: pd.DataFrame, index: int, direction: str) -> dict[str, object]:
        start = max(0, index - self.config.lookback + 1)
        if direction == "short":
            old_extreme = float(scored.loc[start:index, "high"].max())
            structure_level = float(scored.loc[start:index, "low"].min())
        else:
            old_extreme = float(scored.loc[start:index, "low"].min())
            structure_level = float(scored.loc[start:index, "high"].max())
        return {
            "index": int(index),
            "direction": direction,
            "old_extreme": old_extreme,
            "structure_level": structure_level,
            "used": False,
        }

    def _can_trade_second_reversal(self, watch: dict[str, object] | None, index: int) -> bool:
        if watch is None or bool(watch["used"]):
            return False
        return 0 < index - int(watch["index"]) <= self.config.lookback

    def _old_extreme_test_failed(self, scored: pd.DataFrame, watch: dict[str, object], index: int) -> bool:
        watch_index = int(watch["index"])
        if index <= watch_index:
            return False
        window = scored.loc[watch_index + 1 : index]
        old_extreme = float(watch["old_extreme"])
        tolerance = self.config.old_extreme_tolerance_pct
        if str(watch["direction"]) == "short":
            tested = window["high"].astype(float) >= old_extreme * (1.0 - tolerance)
            failed = window["close"].astype(float) < old_extreme
        else:
            tested = window["low"].astype(float) <= old_extreme * (1.0 + tolerance)
            failed = window["close"].astype(float) > old_extreme
        return bool((tested & failed).any())

    def _structure_confirmed(self, row: object, watch: dict[str, object]) -> bool:
        structure_level = float(watch["structure_level"])
        if str(watch["direction"]) == "short":
            return float(row["close"]) < structure_level
        return float(row["close"]) > structure_level

    def _second_reversal_confirmed(self, old_extreme_failed: bool, structure_confirmed: bool) -> bool:
        if self.config.require_old_extreme_test and not old_extreme_failed:
            return False
        if self.config.require_structure_confirmation and not structure_confirmed:
            return False
        return True

    def _metadata(
        self,
        row: object,
        watch: dict[str, object] | None,
        old_extreme_failed: bool,
        structure_confirmed: bool,
    ) -> dict[str, object]:
        return {
            "slope_z": float(row["slope_z"]),
            "old_extreme": float(watch["old_extreme"]) if watch is not None else None,
            "structure_level": float(watch["structure_level"]) if watch is not None else None,
            "old_extreme_test_failed": bool(old_extreme_failed),
            "structure_confirmed": bool(structure_confirmed),
        }

    def _bull_follow_through(self, row: object) -> bool:
        return float(row["close_pos"]) >= self.config.strong_close_pos and float(row["close"]) > float(row["open"])

    def _bull_follow_through_mask(self, scored: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(scored["close_pos"], errors="coerce").ge(self.config.strong_close_pos) & pd.to_numeric(
            scored["close"], errors="coerce"
        ).gt(pd.to_numeric(scored["open"], errors="coerce"))

    def _bear_follow_through(self, row: object) -> bool:
        return (
            float(row["close_pos"]) <= 1.0 - self.config.strong_close_pos and float(row["close"]) < float(row["open"])
        )

    def _bear_follow_through_mask(self, scored: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(scored["close_pos"], errors="coerce").le(
            1.0 - self.config.strong_close_pos
        ) & pd.to_numeric(scored["close"], errors="coerce").lt(pd.to_numeric(scored["open"], errors="coerce"))
