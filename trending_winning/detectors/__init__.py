"""Independent market-pattern detectors."""

from trending_winning.detectors.base import DETECTOR_EVENT_COLUMNS, DetectorEvent, empty_events, validate_detector_events
from trending_winning.detectors.channel import (
    ChannelDetector,
    ChannelDetectorConfig,
    attach_log_regression_channel,
    attach_swing_trend_channel,
)
from trending_winning.detectors.range import RangeDetector, RangeDetectorConfig
from trending_winning.detectors.reversal import ReversalDetector, ReversalDetectorConfig
from trending_winning.detectors.structure import StructureConfig, attach_market_structure
from trending_winning.detectors.trend import TrendDetector, TrendDetectorConfig

__all__ = [
    "DETECTOR_EVENT_COLUMNS",
    "ChannelDetector",
    "ChannelDetectorConfig",
    "DetectorEvent",
    "RangeDetector",
    "RangeDetectorConfig",
    "ReversalDetector",
    "ReversalDetectorConfig",
    "TrendDetector",
    "TrendDetectorConfig",
    "StructureConfig",
    "attach_log_regression_channel",
    "attach_swing_trend_channel",
    "attach_market_structure",
    "empty_events",
    "validate_detector_events",
]
