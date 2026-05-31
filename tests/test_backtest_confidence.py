from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.confidence import (
    CONFIDENCE_Z_SCORE,
    positive_expectancy_probability,
    sample_confidence_statistics,
    standard_error,
    wilson_score_interval,
)


def test_sample_confidence_statistics_reports_trade_return_uncertainty() -> None:
    returns = pd.Series([0.05, -0.02, 0.04, -0.01])

    stats = sample_confidence_statistics(returns)

    win_rate = 0.5
    z2 = CONFIDENCE_Z_SCORE**2
    wilson_denominator = 1.0 + z2 / len(returns)
    wilson_center = win_rate + z2 / (2.0 * len(returns))
    wilson_margin = CONFIDENCE_Z_SCORE * (
        win_rate * (1.0 - win_rate) / len(returns) + z2 / (4.0 * len(returns) ** 2)
    ) ** 0.5
    expected_se = returns.std(ddof=1) / (len(returns) ** 0.5)

    assert stats["win_rate_ci_lower"] == pytest.approx((wilson_center - wilson_margin) / wilson_denominator)
    assert stats["win_rate_ci_upper"] == pytest.approx((wilson_center + wilson_margin) / wilson_denominator)
    assert stats["avg_return_standard_error"] == pytest.approx(expected_se)
    assert stats["avg_return_ci_lower"] == pytest.approx(returns.mean() - CONFIDENCE_Z_SCORE * expected_se)
    assert stats["avg_return_ci_upper"] == pytest.approx(returns.mean() + CONFIDENCE_Z_SCORE * expected_se)
    assert stats["positive_expectancy_probability"] > 0.5


def test_sample_confidence_statistics_uses_neutral_empty_values() -> None:
    stats = sample_confidence_statistics(pd.Series(dtype=float))

    assert stats == {
        "win_rate_ci_lower": 0.0,
        "win_rate_ci_upper": 0.0,
        "avg_return_standard_error": 0.0,
        "avg_return_ci_lower": 0.0,
        "avg_return_ci_upper": 0.0,
        "positive_expectancy_probability": 0.0,
    }


def test_confidence_helpers_handle_small_samples_and_bounds() -> None:
    lower, upper = wilson_score_interval(1.0, 2)

    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert standard_error(pd.Series([0.03])) == 0.0
    assert positive_expectancy_probability(0.01, 0.0) == 1.0
    assert positive_expectancy_probability(-0.01, 0.0) == 0.0
    assert positive_expectancy_probability(0.0, 0.0) == 0.5
