from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


DETECTOR_EVENT_COLUMNS = [
    "event_id",
    "detector_name",
    "stock_code",
    "timeframe",
    "date",
    "bar_index",
    "event_type",
    "direction",
    "signal_price",
    "entry_price",
    "stop_price",
    "confidence",
    "metadata",
]


@dataclass(frozen=True)
class DetectorEvent:
    """模式识别层的标准事件；策略层只读取这些字段生成订单。"""

    event_id: str
    detector_name: str
    stock_code: str
    timeframe: str
    date: pd.Timestamp
    bar_index: int
    event_type: str
    direction: str
    signal_price: float
    entry_price: float
    stop_price: float
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def as_record(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "detector_name": self.detector_name,
            "stock_code": self.stock_code,
            "timeframe": self.timeframe,
            "date": self.date,
            "bar_index": self.bar_index,
            "event_type": self.event_type,
            "direction": self.direction,
            "signal_price": self.signal_price,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }


class Detector(Protocol):
    """所有识别器的协议；趋势、区间、通道、反转彼此独立实现。"""

    name: str

    def detect(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        ...


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(DETECTOR_EVENT_COLUMNS))


def events_to_frame(events: list[DetectorEvent]) -> pd.DataFrame:
    if not events:
        return empty_events()
    return pd.DataFrame([event.as_record() for event in events], columns=DETECTOR_EVENT_COLUMNS)


def validate_detector_events(events: pd.DataFrame, *, detector_name: str = "") -> pd.DataFrame:
    """校验 detector 标准事件契约；策略层只消费通过校验的标准字段。"""
    if not isinstance(events, pd.DataFrame):
        raise ValueError("detector 事件必须是 pandas DataFrame。")
    if events.empty:
        return empty_events()
    missing = [column for column in DETECTOR_EVENT_COLUMNS if column not in events.columns]
    if missing:
        prefix = f"{detector_name} " if detector_name else ""
        raise ValueError(f"{prefix}detector 事件缺少字段：{', '.join(missing)}。")
    invalid_metadata = ~events["metadata"].map(lambda value: isinstance(value, Mapping))
    if invalid_metadata.any():
        prefix = f"{detector_name} " if detector_name else ""
        raise ValueError(f"{prefix}detector 事件 metadata 必须是 dict。")
    return events.loc[:, DETECTOR_EVENT_COLUMNS].copy()
