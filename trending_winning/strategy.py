from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trending_winning.signals.channel import ChannelConfig, attach_trend_channel
from trending_winning.signals.landmark import LandmarkConfig, detect_landmark_candles
from trending_winning.signals.trigger import TriggerConfig, detect_breakout_triggers


@dataclass(frozen=True)
class StrategyConfig:
    """旧版扫描参数集合；串联标志 K、通道和突破触发三步。"""

    landmark_lookback: int = 20
    landmark_range_multiple: float = 1.8
    landmark_volume_multiple: float = 1.8
    landmark_min_body_ratio: float = 0.5
    channel_lookback: int = 40
    channel_min_slope: float = 0.0
    channel_band_atr_multiple: float = 1.0
    trigger_close_buffer_pct: float = 0.0
    trigger_volume_multiple: float = 1.5
    trigger_volume_lookback: int = 20
    require_landmark_trigger: bool = True


def scan_bars(bars: pd.DataFrame, config: StrategyConfig | None = None) -> pd.DataFrame:
    cfg = config or StrategyConfig()
    landmarked = detect_landmark_candles(
        bars,
        LandmarkConfig(
            lookback=cfg.landmark_lookback,
            range_multiple=cfg.landmark_range_multiple,
            volume_multiple=cfg.landmark_volume_multiple,
            min_body_ratio=cfg.landmark_min_body_ratio,
        ),
    )
    channeled = attach_trend_channel(
        landmarked,
        ChannelConfig(
            lookback=cfg.channel_lookback,
            min_slope=cfg.channel_min_slope,
            band_atr_multiple=cfg.channel_band_atr_multiple,
        ),
    )
    return detect_breakout_triggers(
        channeled,
        TriggerConfig(
            close_buffer_pct=cfg.trigger_close_buffer_pct,
            volume_multiple=cfg.trigger_volume_multiple,
            volume_lookback=cfg.trigger_volume_lookback,
            require_landmark=cfg.require_landmark_trigger,
        ),
    )
