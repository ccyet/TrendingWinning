from __future__ import annotations

import numpy as np

from trending_winning.backtest.trailing_take_profit import trailing_take_profit_masks


def test_trailing_take_profit_module_tracks_long_from_completed_profit_bar() -> None:
    opens = np.array([10.0, 10.8, 10.8])
    highs = np.array([10.2, 11.0, 10.9])
    lows = np.array([9.9, 10.7, 10.6])
    liquid = np.array([True, True, True])

    result = trailing_take_profit_masks(
        opens,
        highs,
        lows,
        liquid,
        side="long",
        entry_price=10.0,
        activation_pct=0.05,
        drawdown_pct=0.03,
    )

    assert result.gap.tolist() == [False, False, False]
    assert result.hit.tolist() == [False, False, True]
    assert result.armed.tolist() == [False, False, True]
    assert result.prices.tolist() == [9.7, 9.894, 10.67]


def test_trailing_take_profit_module_tracks_short_from_completed_profit_bar() -> None:
    opens = np.array([10.0, 9.2, 9.2])
    highs = np.array([10.1, 9.3, 9.35])
    lows = np.array([9.8, 9.0, 9.1])
    liquid = np.array([True, True, True])

    result = trailing_take_profit_masks(
        opens,
        highs,
        lows,
        liquid,
        side="short",
        entry_price=10.0,
        activation_pct=0.05,
        drawdown_pct=0.03,
    )

    assert result.gap.tolist() == [False, False, False]
    assert result.hit.tolist() == [False, False, True]
    assert result.armed.tolist() == [False, False, True]
    assert result.prices.tolist() == [10.3, 10.094, 9.27]
