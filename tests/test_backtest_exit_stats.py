from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.exit_stats import summarize_exit_reasons


def test_exit_reason_summary_reports_counts_rates_and_other_bucket() -> None:
    trades = pd.DataFrame(
        {
            "exit_reason": [
                "take_profit",
                "trailing_take_profit",
                "trailing_take_profit",
                "stop_loss",
                "custom_exit",
                "",
            ]
        }
    )

    stats = summarize_exit_reasons(trades)

    assert stats["take_profit_exit_count"] == 1.0
    assert stats["take_profit_exit_rate"] == pytest.approx(1 / 6)
    assert stats["trailing_take_profit_exit_count"] == 2.0
    assert stats["trailing_take_profit_exit_rate"] == pytest.approx(2 / 6)
    assert stats["primary_exit_reason"] == "trailing_take_profit"
    assert stats["primary_exit_reason_count"] == 2.0
    assert stats["primary_exit_reason_rate"] == pytest.approx(2 / 6)
    assert stats["stop_loss_exit_count"] == 1.0
    assert stats["stop_loss_exit_rate"] == pytest.approx(1 / 6)
    assert stats["other_exit_count"] == 2.0
    assert stats["other_exit_rate"] == pytest.approx(2 / 6)


def test_exit_reason_summary_returns_stable_empty_keys() -> None:
    stats = summarize_exit_reasons(pd.DataFrame())

    assert stats["take_profit_exit_count"] == 0.0
    assert stats["trailing_take_profit_exit_rate"] == 0.0
    assert stats["stop_loss_exit_count"] == 0.0
    assert stats["max_holding_exit_rate"] == 0.0
    assert stats["end_of_data_exit_count"] == 0.0
    assert stats["other_exit_rate"] == 0.0
    assert stats["primary_exit_reason"] == ""
    assert stats["primary_exit_reason_count"] == 0.0
    assert stats["primary_exit_reason_rate"] == 0.0
