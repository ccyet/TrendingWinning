from __future__ import annotations

from inspect import getsource

import pandas as pd

from trending_winning.detectors.trend import TrendDetector, TrendDetectorConfig, _count_pullback_legs


def _bars(closes: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, close in enumerate(closes):
        previous = closes[index - 1] if index else close
        up_bar = close >= previous
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.18 if up_bar else close + 0.18,
                "high": close + (0.05 if up_bar else 0.25),
                "low": close - (0.25 if up_bar else 0.05),
                "close": close,
                "volume": 1000.0 + index,
                "amount": close * (1000.0 + index),
            }
        )
    return pd.DataFrame(rows)


def _reference_pullback_legs(close: pd.Series, *, direction: str, lookback: int) -> list[int]:
    values = close.astype(float).to_list()
    out: list[int] = []
    for index in range(len(values)):
        start = max(1, index - lookback)
        legs = 0
        in_leg = False
        for cursor in range(start, index):
            delta = values[cursor] - values[cursor - 1]
            is_pullback = delta < 0 if direction == "bull" else delta > 0
            if is_pullback and not in_leg:
                legs += 1
                in_leg = True
            elif not is_pullback:
                in_leg = False
        out.append(legs)
    return out


def test_pullback_leg_counter_matches_reference_with_window_boundary_continuation() -> None:
    closes = pd.Series([10.0, 9.9, 9.8, 10.0, 9.7, 9.6, 9.9, 9.5, 9.4])

    actual = _count_pullback_legs(closes, direction="bull", lookback=3)

    assert actual.tolist() == _reference_pullback_legs(closes, direction="bull", lookback=3)


def test_pullback_leg_counter_uses_prefix_scan_not_nested_cursor_loop() -> None:
    source = getsource(_count_pullback_legs)

    assert "for cursor in range" not in source


def test_trend_detector_labels_bull_h2_after_two_legged_pullback() -> None:
    bars = _bars([10.0, 10.25, 10.5, 10.75, 11.0, 10.82, 10.98, 10.78, 11.15, 11.45, 11.75])

    events = TrendDetector(TrendDetectorConfig(lookback=4, min_trend_score=0.15, pullback_lookback=5)).detect(
        bars,
        timeframe="30m",
    )

    h2 = events.loc[events["event_type"] == "bull_h2_setup"]
    assert not h2.empty
    last = h2.iloc[-1]
    assert last["direction"] == "long"
    assert last["entry_price"] > last["signal_price"]
    assert last["metadata"]["trend_state"] == "bull"
    assert last["metadata"]["pullback_legs"] >= 2


def test_trend_detector_labels_bear_l2_after_two_legged_pullback() -> None:
    bars = _bars([12.0, 11.75, 11.5, 11.25, 11.0, 11.18, 11.02, 11.22, 10.85, 10.55, 10.25])

    events = TrendDetector(TrendDetectorConfig(lookback=4, min_trend_score=0.15, pullback_lookback=5)).detect(
        bars,
        timeframe="30m",
    )

    l2 = events.loc[events["event_type"] == "bear_l2_setup"]
    assert not l2.empty
    last = l2.iloc[-1]
    assert last["direction"] == "short"
    assert last["entry_price"] < last["signal_price"]
    assert last["metadata"]["trend_state"] == "bear"
    assert last["metadata"]["pullback_legs"] >= 2
