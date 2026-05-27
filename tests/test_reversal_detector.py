from __future__ import annotations

import pandas as pd

from trending_winning.detectors.reversal import ReversalDetector, ReversalDetectorConfig


def _bear_reversal_bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [10.0, 10.5, 11.0, 11.5, 11.0, 11.45, 10.65, 10.1, 9.8]
    for index, close in enumerate(closes):
        bearish = index in {4, 6, 7, 8}
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close + 0.35 if bearish else close - 0.1,
                "high": close + 0.4,
                "low": close - 0.1 if bearish else close - 0.2,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _bear_reversal_without_failed_old_high_test() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = [10.0, 10.5, 11.0, 11.5, 11.0, 10.75, 10.35, 10.1]
    for index, close in enumerate(closes):
        bearish = index >= 4
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close + 0.35 if bearish else close - 0.1,
                "high": close + 0.25,
                "low": close - 0.1 if bearish else close - 0.2,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    return pd.DataFrame(rows)


def test_reversal_detector_requires_failed_old_extreme_and_structure_confirmation_for_second_trade() -> None:
    events = ReversalDetector(ReversalDetectorConfig(lookback=3, old_extreme_tolerance_pct=0.02)).detect(
        _bear_reversal_bars(),
        timeframe="30m",
    )

    assert events["event_type"].tolist()[:2] == ["first_reversal_watch_short", "second_reversal_short"]
    assert events.loc[events["event_type"] == "first_reversal_watch_short", "direction"].tolist() == ["neutral"]
    assert events.loc[events["event_type"] == "second_reversal_short", "direction"].tolist() == ["short"]
    second = events.loc[events["event_type"] == "second_reversal_short"].iloc[0]
    assert second["metadata"]["old_extreme_test_failed"] is True
    assert second["metadata"]["structure_confirmed"] is True


def test_reversal_detector_does_not_trade_second_reversal_without_failed_old_extreme_test() -> None:
    events = ReversalDetector(ReversalDetectorConfig(lookback=3, old_extreme_tolerance_pct=0.02)).detect(
        _bear_reversal_without_failed_old_high_test(),
        timeframe="30m",
    )

    assert "first_reversal_watch_short" in events["event_type"].tolist()
    assert "second_reversal_short" not in events["event_type"].tolist()
