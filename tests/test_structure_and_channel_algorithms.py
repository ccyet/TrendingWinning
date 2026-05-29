from __future__ import annotations

from inspect import getsource

import numpy as np
import pytest
import pandas as pd

from trending_winning.detectors.channel import (
    ChannelDetector,
    ChannelDetectorConfig,
    _attach_group_swing_channel,
    attach_log_regression_channel,
    attach_swing_trend_channel,
)
from trending_winning.detectors.structure import (
    StructureConfig,
    attach_market_structure,
)
from trending_winning.detectors.pivots import confirmed_pivots


def _bars(closes: list[float]) -> pd.DataFrame:
    rows = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.05,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _reference_confirmed_pivots(
    high: np.ndarray,
    low: np.ndarray,
    left_bars: int,
    right_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    pivot_high = np.full(len(high), False)
    pivot_low = np.full(len(low), False)
    for index in range(left_bars, len(high) - right_bars):
        high_window = high[index - left_bars : index + right_bars + 1]
        low_window = low[index - left_bars : index + right_bars + 1]
        pivot_high[index] = bool(high[index] == np.nanmax(high_window) and high[index] > np.nanmax(high_window[:left_bars]))
        pivot_low[index] = bool(low[index] == np.nanmin(low_window) and low[index] < np.nanmin(low_window[:left_bars]))
    return pivot_high, pivot_low


def test_confirmed_pivots_match_reference_swing_rules() -> None:
    high = np.array([10.2, 11.2, 10.6, 12.4, 11.4, 13.3, 12.2, 13.9, 13.6])
    low = np.array([9.8, 10.8, 10.2, 12.0, 11.0, 12.9, 11.8, 13.4, 13.2])

    actual_high, actual_low = confirmed_pivots(high, low, left_bars=1, right_bars=1)
    expected_high, expected_low = _reference_confirmed_pivots(high, low, 1, 1)

    assert actual_high.tolist() == expected_high.tolist()
    assert actual_low.tolist() == expected_low.tolist()


def test_confirmed_pivots_uses_vectorized_window_scan() -> None:
    source = getsource(confirmed_pivots)

    assert "for index in range" not in source


def test_market_structure_marks_pivots_and_structure_breaks_from_confirmed_swings() -> None:
    bars = _bars([10.0, 11.0, 10.4, 12.2, 11.2, 13.1, 12.0, 13.8, 13.4])

    structured = attach_market_structure(bars, StructureConfig(left_bars=1, right_bars=1, break_buffer=0.0))

    assert structured["pivot_high"].sum() >= 3
    assert structured["pivot_low"].sum() >= 2
    assert structured.loc[3, "structure_label"] == "HH"
    assert structured.loc[4, "structure_label"] == "HL"
    assert bool(structured.loc[5, "bos_up"]) is True
    assert structured.loc[5, "last_swing_high"] < structured.loc[5, "close"]


def test_channel_detector_uses_log_regression_channel_and_emits_standard_events() -> None:
    bars = _bars([10.0, 10.4, 10.8, 11.3, 11.8, 12.3, 12.9, 13.6, 15.8])

    events = ChannelDetector(ChannelDetectorConfig(lookback=6, sigma_multiple=1.2)).detect(bars)

    assert not events.empty
    assert set(events["detector_name"]) == {"channel"}
    assert "channel_upper" in events.iloc[-1]["metadata"]
    assert "channel_r2" in events.iloc[-1]["metadata"]
    assert events.iloc[-1]["event_type"] in {"channel_overshoot_up", "channel_break_down"}


def test_channel_breakout_enters_above_or_below_signal_bar_extreme() -> None:
    bars = _bars([10.0, 10.4, 10.8, 11.3, 11.8, 12.3, 12.9, 13.6, 15.8])

    event = ChannelDetector(ChannelDetectorConfig(lookback=6, sigma_multiple=1.2)).detect(bars).iloc[-1]
    signal_bar = bars.iloc[int(event["bar_index"])]

    assert event["direction"] == "long"
    assert event["signal_price"] == pytest.approx(signal_bar["high"])
    assert event["entry_price"] == pytest.approx(signal_bar["high"] + 0.01)


def test_channel_breakdown_enters_below_signal_bar_extreme() -> None:
    bars = _bars([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 8.0])

    event = ChannelDetector(ChannelDetectorConfig(lookback=6, sigma_multiple=1.2)).detect(bars).iloc[-1]
    signal_bar = bars.iloc[int(event["bar_index"])]

    assert event["direction"] == "short"
    assert event["signal_price"] == pytest.approx(signal_bar["low"])
    assert event["entry_price"] == pytest.approx(signal_bar["low"] - 0.01)


def test_channel_detector_requires_prior_channel_boundary_for_breakout() -> None:
    bars = _bars([10.0, 10.0, 15.0])

    events = ChannelDetector(ChannelDetectorConfig(lookback=3, sigma_multiple=0.1)).detect(bars)

    assert events.empty


def test_log_regression_channel_reports_fit_quality_and_lagged_boundaries() -> None:
    bars = _bars([10.0 * (1.02**index) for index in range(12)])

    channeled = attach_log_regression_channel(bars, ChannelDetectorConfig(lookback=5, sigma_multiple=1.0))

    fitted = channeled.dropna(subset=["channel_r2"])
    assert not fitted.empty
    assert fitted["channel_r2"].min() == pytest.approx(1.0)
    assert channeled.loc[6, "prev_channel_upper"] == pytest.approx(channeled.loc[5, "channel_upper"])
    assert channeled.loc[6, "prev_channel_lower"] == pytest.approx(channeled.loc[5, "channel_lower"])


def test_swing_channel_uses_confirmed_pivots_as_trendline_anchors() -> None:
    bars = _bars([10.0, 11.0, 10.4, 12.2, 11.2, 13.1, 12.0, 13.8, 14.4])

    channeled = attach_swing_trend_channel(
        bars,
        ChannelDetectorConfig(channel_method="swing", swing_left_bars=1, swing_right_bars=1),
    )
    last = channeled.iloc[-1]

    assert pd.notna(last["channel_upper"])
    assert pd.notna(last["channel_lower"])
    assert last["channel_method"] == "swing"
    assert last["channel_anchor_index_1"] < last["channel_anchor_index_2"] < last.name


def test_channel_detector_can_emit_events_from_swing_channel() -> None:
    bars = _bars([10.0, 11.0, 10.4, 12.2, 11.2, 13.1, 12.0, 13.8, 15.2])

    events = ChannelDetector(
        ChannelDetectorConfig(channel_method="swing", swing_left_bars=1, swing_right_bars=1)
    ).detect(bars, timeframe="30m")

    assert not events.empty
    assert events.iloc[-1]["metadata"]["channel_method"] == "swing"
    assert "channel_anchor_index_1" in events.iloc[-1]["metadata"]


def test_channel_detector_emits_events_without_row_record_scan() -> None:
    source = getsource(ChannelDetector.detect)

    assert ".to_records(" not in source


def test_swing_channel_maintains_confirmed_anchors_incrementally() -> None:
    source = getsource(_attach_group_swing_channel)

    assert "np.flatnonzero" not in source
    assert "_line_values(" not in source
