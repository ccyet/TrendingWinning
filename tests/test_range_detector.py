from __future__ import annotations

import pandas as pd

from trending_winning.detectors.range import RangeDetector, RangeDetectorConfig


def _bars(closes: list[float], *, half_range: float = 0.4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.05,
                "high": close + half_range,
                "low": close - half_range,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_range_detector_outputs_mathematical_range_score_metadata() -> None:
    bars = _bars([10.0, 10.3, 10.1, 10.35, 10.15, 10.4, 10.2, 10.35, 10.15, 10.3])

    events = RangeDetector(RangeDetectorConfig(lookback=5)).detect(bars, timeframe="30m")

    middle = events.loc[events["event_type"] == "no_trade_middle"]
    assert not middle.empty
    metadata = middle.iloc[-1]["metadata"]
    assert {"range_score", "overlap_mean", "ema_flatness", "directional_efficiency"}.issubset(metadata)
    assert metadata["range_score"] >= 0.8


def test_range_detector_does_not_mark_strong_trend_middle_as_range_noise() -> None:
    trend_closes = [10.0 + index * 0.25 for index in range(14)]

    events = RangeDetector(RangeDetectorConfig(lookback=5)).detect(_bars(trend_closes, half_range=1.0), timeframe="30m")

    assert "no_trade_middle" not in set(events["event_type"])
