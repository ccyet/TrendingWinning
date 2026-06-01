from __future__ import annotations

from trending_winning.backtest.reason_labels import (
    DATA_ISSUE_LABELS,
    EXIT_REASON_LABELS,
    ORDER_REJECT_REASON_LABELS,
    STRATEGY_FILTER_REASON_LABELS,
    exit_reason_label,
    reason_label,
    reason_label_with_code,
)


def test_reason_label_source_covers_order_strategy_data_and_exit_codes() -> None:
    assert ORDER_REJECT_REASON_LABELS["actual_risk_too_high"] == "止损风险过大"
    assert STRATEGY_FILTER_REASON_LABELS["higher_timeframe_stale"] == "大周期信号过旧"
    assert STRATEGY_FILTER_REASON_LABELS["terminal_false_breakout_risk"] == "末端假突破风险"
    assert DATA_ISSUE_LABELS["data_coverage_below_min"] == "覆盖率低于门槛"
    assert EXIT_REASON_LABELS["trailing_take_profit"] == "回撤止盈"


def test_reason_label_helpers_keep_unknown_codes_and_traceable_known_codes() -> None:
    assert reason_label("actual_risk_too_high") == "止损风险过大"
    assert reason_label("higher_timeframe_stale") == "大周期信号过旧"
    assert reason_label("custom_reason") == "custom_reason"
    assert reason_label_with_code("actual_risk_too_high") == "止损风险过大（actual_risk_too_high）"
    assert reason_label_with_code("custom_reason") == "custom_reason"
    assert exit_reason_label("trailing_take_profit") == "回撤止盈"
    assert exit_reason_label("custom_exit") == "custom_exit"
