from __future__ import annotations

from inspect import getsource

import numpy as np
import pandas as pd

from trending_winning.signals.channel import ChannelConfig, _attach_group_channel, attach_trend_channel
from trending_winning.signals.landmark import LandmarkConfig, detect_landmark_candles
from trending_winning.signals.trigger import TriggerConfig, detect_breakout_triggers


def _sample_bars() -> pd.DataFrame:
    close = [10, 10.2, 10.4, 10.7, 10.9, 11.2, 11.5, 11.7, 11.9, 12.2, 12.4, 13.8]
    rows = []
    for index, value in enumerate(close):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": value - (0.1 if index != 11 else 1.1),
                "high": value + (0.2 if index != 11 else 0.4),
                "low": value - (0.2 if index != 11 else 1.2),
                "close": value,
                "volume": 1000.0 if index != 11 else 3200.0,
                "amount": value * (1000.0 if index != 11 else 3200.0),
            }
        )
    return pd.DataFrame(rows)


def test_detect_landmark_candles_marks_large_volume_body_breakout_bar() -> None:
    out = detect_landmark_candles(
        _sample_bars(),
        LandmarkConfig(lookback=6, range_multiple=1.8, volume_multiple=2.0, min_body_ratio=0.55),
    )

    assert out["is_landmark"].tolist()[-1] is True
    assert out["landmark_reason"].tolist()[-1] == "range+volume+body"


def test_channel_and_trigger_use_previous_channel_boundary() -> None:
    bars = _sample_bars()
    channel = attach_trend_channel(bars, ChannelConfig(lookback=8, min_slope=0.05, band_atr_multiple=1.0))
    triggered = detect_breakout_triggers(
        channel,
        TriggerConfig(close_buffer_pct=0.01, volume_multiple=1.5, require_landmark=False),
    )

    last = triggered.iloc[-1]
    previous = triggered.iloc[-2]
    assert last["channel_direction"] == "up"
    assert bool(previous["breakout_trigger"]) is False
    assert bool(last["breakout_trigger"]) is True
    assert last["trigger_price"] == last["close"]


def _reference_channel(group: pd.DataFrame, cfg: ChannelConfig) -> pd.DataFrame:
    result = group.copy()
    highs = pd.to_numeric(result["high"], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(result["low"], errors="coerce").astype(float).to_numpy()
    closes = pd.to_numeric(result["close"], errors="coerce").astype(float).to_numpy()
    true_range = highs - lows
    slopes = np.full(len(result), np.nan)
    mids = np.full(len(result), np.nan)
    uppers = np.full(len(result), np.nan)
    lowers = np.full(len(result), np.nan)
    widths = np.full(len(result), np.nan)
    x_values = np.arange(cfg.lookback, dtype=float)
    for index in range(cfg.lookback - 1, len(result)):
        start = index - cfg.lookback + 1
        close_window = closes[start : index + 1]
        high_window = highs[start : index + 1]
        low_window = lows[start : index + 1]
        if not np.isfinite(close_window).all():
            continue
        slope, intercept = np.polyfit(x_values, close_window, 1)
        fitted = intercept + slope * x_values
        mid = float(fitted[-1])
        residual_band = max(float(np.nanmax(high_window - fitted)), float(np.nanmax(fitted - low_window)), 0.0)
        atr_band = float(np.nanmean(true_range[start : index + 1])) * cfg.band_atr_multiple
        band = max(residual_band, atr_band)
        slopes[index] = float(slope)
        mids[index] = mid
        uppers[index] = mid + band
        lowers[index] = mid - band
        widths[index] = band * 2.0
    result["channel_slope"] = slopes
    result["channel_mid"] = mids
    result["channel_upper"] = uppers
    result["channel_lower"] = lowers
    result["channel_width"] = widths
    result["channel_direction"] = np.select(
        [result["channel_slope"] > cfg.min_slope, result["channel_slope"] < -cfg.min_slope],
        ["up", "down"],
        default="flat",
    )
    return result


def test_legacy_channel_matches_reference_regression() -> None:
    bars = _sample_bars()
    cfg = ChannelConfig(lookback=5, min_slope=0.03, band_atr_multiple=1.2)

    actual = _attach_group_channel(bars, cfg)
    expected = _reference_channel(bars, cfg)

    columns = ["channel_slope", "channel_mid", "channel_upper", "channel_lower", "channel_width"]
    np.testing.assert_allclose(actual[columns].to_numpy(), expected[columns].to_numpy(), equal_nan=True)
    assert actual["channel_direction"].tolist() == expected["channel_direction"].tolist()


def test_legacy_channel_uses_vectorized_regression_not_polyfit_loop() -> None:
    source = getsource(_attach_group_channel)

    assert "np.polyfit" not in source
    assert "for index in range" not in source
