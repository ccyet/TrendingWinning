from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trending_winning.detectors.base import DetectorEvent, empty_events, events_to_frame
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

        events: list[DetectorEvent] = []
        for symbol, group in featured.groupby("stock_code", sort=False):
            scored = group.reset_index(drop=True).copy()
            scored["slope_z"] = rolling_slope_z(scored["close"], self.config.lookback).fillna(0.0)
            scored["_prior_slope"] = scored["slope_z"].shift(1)
            bull_watch: dict[str, object] | None = None
            bear_watch: dict[str, object] | None = None
            for index, row in enumerate(scored.to_records(index=False)):
                event_type = ""
                direction = "neutral"
                metadata = self._metadata(row, None, old_extreme_failed=False, structure_confirmed=False)
                if self._first_bull_reversal(row):
                    event_type = "first_reversal_watch_long"
                    direction = "neutral"
                    bull_watch = self._new_watch(scored, index, "long")
                    metadata = self._metadata(row, bull_watch, old_extreme_failed=False, structure_confirmed=False)
                elif self._first_bear_reversal(row):
                    event_type = "first_reversal_watch_short"
                    direction = "neutral"
                    bear_watch = self._new_watch(scored, index, "short")
                    metadata = self._metadata(row, bear_watch, old_extreme_failed=False, structure_confirmed=False)
                elif self._can_trade_second_reversal(bull_watch, index) and self._bull_follow_through(row):
                    old_extreme_failed = self._old_extreme_test_failed(scored, bull_watch, index)
                    structure_confirmed = self._structure_confirmed(row, bull_watch)
                    if not self._second_reversal_confirmed(old_extreme_failed, structure_confirmed):
                        continue
                    event_type = "second_reversal_long"
                    direction = "long"
                    bull_watch["used"] = True
                    metadata = self._metadata(row, bull_watch, old_extreme_failed, structure_confirmed)
                elif self._can_trade_second_reversal(bear_watch, index) and self._bear_follow_through(row):
                    old_extreme_failed = self._old_extreme_test_failed(scored, bear_watch, index)
                    structure_confirmed = self._structure_confirmed(row, bear_watch)
                    if not self._second_reversal_confirmed(old_extreme_failed, structure_confirmed):
                        continue
                    event_type = "second_reversal_short"
                    direction = "short"
                    bear_watch["used"] = True
                    metadata = self._metadata(row, bear_watch, old_extreme_failed, structure_confirmed)
                if not event_type:
                    continue
                symbol_text = str(symbol)
                entry, stop, signal = self._event_prices(row, direction, event_type)
                events.append(
                    DetectorEvent(
                        event_id=f"{self.name}:{symbol_text}:{pd.Timestamp(row['date']).isoformat()}:{event_type}",
                        detector_name=self.name,
                        stock_code=symbol_text,
                        timeframe=timeframe,
                        date=pd.Timestamp(row["date"]),
                        bar_index=int(index),
                        event_type=event_type,
                        direction=direction,
                        signal_price=signal,
                        entry_price=entry,
                        stop_price=stop,
                        confidence=0.5 if direction == "neutral" else 1.0,
                        metadata=metadata,
                    )
                )
        return events_to_frame(events)

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

    def _first_bear_reversal(self, row: object) -> bool:
        return (
            float(row["_prior_slope"]) > 0
            and float(row["slope_z"]) <= 0
            and float(row["close_pos"]) <= 1.0 - self.config.strong_close_pos
            and float(row["body_ratio"]) >= self.config.min_body_ratio
            and float(row["close"]) < float(row["open"])
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

    def _bear_follow_through(self, row: object) -> bool:
        return (
            float(row["close_pos"]) <= 1.0 - self.config.strong_close_pos and float(row["close"]) < float(row["open"])
        )
