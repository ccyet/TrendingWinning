from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.decision_stats import (
    compute_decision_reason_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)


def test_decision_reason_statistics_reports_global_and_group_rates() -> None:
    decisions = pd.DataFrame(
        {
            "strategy_name": ["trend_signal_bar", "trend_signal_bar", "range_signal_bar", "range_signal_bar"],
            "detector_name": ["trend", "trend", "range", "range"],
            "status": ["accepted", "rejected", "rejected", "rejected"],
            "reason": ["", "no_fill", "no_capital", "no_capital"],
            "actual_risk_pct": [0.03, 0.0, 0.04, 0.06],
            "actual_chase_pct": [0.01, 0.0, 0.02, 0.03],
            "actual_reward_to_risk": [2.0, 0.0, 1.5, 1.0],
        }
    )

    stats = compute_decision_reason_statistics(decisions)
    by_key = stats.set_index(["strategy_name", "detector_name", "status", "reason"])

    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "decision_rate"] == pytest.approx(0.25)
    assert by_key.loc[("trend_signal_bar", "trend", "accepted", ""), "group_decision_rate"] == pytest.approx(0.5)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "decision_count"] == 2
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "group_decision_rate"] == pytest.approx(1.0)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "avg_actual_risk_pct"] == pytest.approx(0.05)
    assert by_key.loc[("range_signal_bar", "range", "rejected", "no_capital"), "min_actual_reward_to_risk"] == pytest.approx(1.0)


def test_order_decision_summary_reports_execution_quality_and_custom_rejections() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["accepted", "rejected", "accepted", "rejected"],
            "reason": ["", "price_limit_blocked", "", "price_limit_blocked"],
            "capital_fraction": [0.25, 0.0, 0.35, 0.0],
            "risk_fraction": [0.01, 0.0, 0.015, 0.0],
            "margin_fraction": [0.25, 0.0, 0.7, 0.0],
            "actual_entry_price": [10.0, 0.0, 20.0, 0.0],
            "actual_risk_pct": [0.03, 0.0, 0.04, 0.0],
            "actual_chase_pct": [0.01, 0.0, 0.02, 0.0],
            "actual_reward_to_risk": [2.0, 0.0, 1.5, 0.0],
        }
    )

    stats = summarize_order_decisions(decisions)

    assert stats["order_count"] == 4.0
    assert stats["acceptance_rate"] == pytest.approx(0.5)
    assert stats["rejected_price_limit_blocked_count"] == 2.0
    assert stats["avg_accepted_capital_fraction"] == pytest.approx(0.3)
    assert stats["accepted_executed_order_count"] == 2.0
    assert stats["avg_accepted_actual_reward_to_risk"] == pytest.approx(1.75)


def test_strategy_filter_summary_reports_standard_and_custom_rejections() -> None:
    decisions = pd.DataFrame(
        {
            "status": ["accepted", "rejected", "rejected"],
            "reason": ["", "higher_timeframe_mismatch", "same_timeframe_middle"],
        }
    )

    stats = summarize_strategy_filter_decisions(decisions)

    assert stats["strategy_signal_count"] == 3.0
    assert stats["strategy_filter_acceptance_rate"] == pytest.approx(1 / 3)
    assert stats["strategy_rejected_higher_timeframe_mismatch_count"] == 1.0
    assert stats["strategy_rejected_same_timeframe_middle_count"] == 1.0
