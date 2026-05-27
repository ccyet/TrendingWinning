from __future__ import annotations

from inspect import getsource

import numpy as np
import pandas as pd

from trending_winning.detectors.features import rolling_slope_z


def _reference_rolling_slope_z(close: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").astype(float).to_numpy()
    out = np.full(len(values), np.nan)
    x = np.arange(lookback, dtype=float)
    x_centered = x - x.mean()
    denominator = float(np.dot(x_centered, x_centered))
    log_values = np.log(np.where(values > 0, values, np.nan))
    returns = pd.Series(log_values).diff().rolling(lookback, min_periods=3).std().to_numpy()
    for index in range(lookback - 1, len(values)):
        window = log_values[index - lookback + 1 : index + 1]
        if not np.isfinite(window).all():
            continue
        y_centered = window - window.mean()
        slope = float(np.dot(x_centered, y_centered) / denominator)
        volatility = returns[index]
        out[index] = 0.0 if not np.isfinite(volatility) or volatility == 0 else slope / float(volatility)
    return pd.Series(out, index=close.index)


def test_rolling_slope_z_matches_reference_with_invalid_prices() -> None:
    close = pd.Series([10.0, 10.1, 10.4, 0.0, 10.8, 11.0, 11.4, 11.3, 11.8, 12.1])

    actual = rolling_slope_z(close, lookback=4)
    expected = _reference_rolling_slope_z(close, lookback=4)

    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), equal_nan=True)
    assert actual.index.equals(close.index)


def test_rolling_slope_z_uses_prefix_formula_not_window_loop() -> None:
    source = getsource(rolling_slope_z)

    assert "for index in range" not in source
