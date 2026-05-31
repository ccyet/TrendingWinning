from __future__ import annotations

import numpy as np
import pandas as pd

from trending_winning.backtest.indicators import completed_bar_moving_average


def test_completed_bar_moving_average_aligns_to_path_index_without_future_data() -> None:
    bars = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0, 15.0, 16.0]},
        index=pd.Index([10, 11, 12, 13, 14]),
    )

    result = completed_bar_moving_average(bars, pd.Index([12, 13, 14]), 3)

    assert result is not None
    np.testing.assert_allclose(result, np.array([np.nan, 11.0, (11.0 + 12.0 + 15.0) / 3]), equal_nan=True)


def test_completed_bar_moving_average_returns_none_when_period_disabled() -> None:
    bars = pd.DataFrame({"close": [10.0, 11.0]})

    assert completed_bar_moving_average(bars, bars.index, 0) is None
    assert completed_bar_moving_average(bars, bars.index, 1) is None
