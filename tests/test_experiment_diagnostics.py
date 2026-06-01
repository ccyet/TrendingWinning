from __future__ import annotations

import pandas as pd

from trending_winning.backtest.experiment_diagnostics import (
    CASE_DIAGNOSTIC_COLUMNS,
    EXPERIMENT_DIAGNOSTIC_COLUMNS,
    case_diagnostic_statistics,
    diagnostic_summary_fields,
    experiment_diagnostic_report,
)


def test_experiment_diagnostic_report_flags_core_risks() -> None:
    stats = {
        "trade_count": 12.0,
        "order_count": 60.0,
        "acceptance_rate": 0.18,
        "primary_rejected_reason": "actual_risk_too_high",
        "primary_rejected_reason_count": 24.0,
        "primary_rejected_reason_rate": 0.5,
        "strategy_filter_rejection_rate": 0.72,
        "primary_strategy_rejected_reason": "terminal_false_breakout_risk",
        "primary_strategy_rejected_reason_count": 18.0,
        "primary_strategy_rejected_reason_rate": 0.42,
        "max_drawdown": -0.24,
        "profit_factor": 0.9,
        "monthly_worst_return": -0.11,
        "avg_mae_r": -0.9,
        "max_margin_exposure": 1.2,
        "data_weighted_coverage_ratio": 0.91,
        "data_coverage_below_min_count": 2.0,
        "primary_data_issue": "data_coverage_below_min",
        "primary_data_issue_count": 2.0,
        "primary_data_issue_rate": 0.4,
        "primary_exit_reason": "stop_loss",
        "primary_exit_reason_count": 7.0,
        "primary_exit_reason_rate": 7 / 12,
    }

    report = experiment_diagnostic_report(stats)

    assert report.columns.tolist() == EXPERIMENT_DIAGNOSTIC_COLUMNS.tolist()
    by_check = report.set_index("check")
    assert by_check.loc["数据覆盖", "status"] == "失败"
    assert "data_coverage_below_min 2 项，占数据问题 40.0%" in by_check.loc["数据覆盖", "detail"]
    assert by_check.loc["交易样本", "status"] == "关注"
    assert by_check.loc["订单接受率", "status"] == "关注"
    assert "actual_risk_too_high 24 笔，占拒单 50.0%" in by_check.loc["订单接受率", "detail"]
    assert by_check.loc["策略过滤", "status"] == "关注"
    assert "terminal_false_breakout_risk 18 条，占过滤拒绝 42.0%" in by_check.loc["策略过滤", "detail"]
    assert by_check.loc["回撤压力", "status"] == "关注"
    assert by_check.loc["收益质量", "status"] == "失败"
    assert by_check.loc["退出结构", "status"] == "关注"
    assert "止损 7 笔，占退出 58.3%" in by_check.loc["退出结构", "detail"]
    assert by_check.loc["路径风险", "status"] == "关注"
    assert by_check.loc["资金暴露", "status"] == "关注"


def test_experiment_diagnostic_report_marks_zero_trade_as_failed() -> None:
    report = experiment_diagnostic_report({"trade_count": 0.0, "order_count": 0.0})

    by_check = report.set_index("check")
    assert by_check.loc["交易样本", "status"] == "失败"
    assert by_check.loc["订单接受率", "status"] == "失败"
    assert by_check.loc["退出结构", "status"] == "失败"
    assert by_check.loc["交易样本", "detail"] == "没有成交，统计结果不能用于评估策略质量。"


def test_experiment_diagnostic_report_uses_data_coverage_when_stats_missing() -> None:
    data_coverage = pd.DataFrame(
        {
            "coverage_ratio": [0.88, 0.96],
            "status": ["coverage_below_min", "ok"],
        }
    )

    report = experiment_diagnostic_report({"trade_count": 35.0}, data_coverage=data_coverage)

    row = report.set_index("check").loc["数据覆盖"]
    assert row["status"] == "失败"
    assert row["value"] == 0.88


def test_diagnostic_summary_fields_counts_case_statuses() -> None:
    summary = diagnostic_summary_fields(
        {
            "trade_count": 12.0,
            "order_count": 60.0,
            "acceptance_rate": 0.18,
            "profit_factor": 0.9,
            "data_weighted_coverage_ratio": 0.91,
            "data_coverage_below_min_count": 1.0,
        }
    )

    assert summary["diagnostic_failed_count"] == 2.0
    assert summary["diagnostic_attention_count"] == 2.0
    assert summary["diagnostic_max_severity"] == 2.0
    assert summary["diagnostic_primary_issue"] == "数据覆盖"


def test_case_diagnostic_statistics_preserves_sweep_rank_context() -> None:
    table = pd.DataFrame(
        {
            "sweep_rank": [1],
            "pareto_rank": [1],
            "is_pareto_efficient": [True],
            "case_name": ["case-001"],
            "case_config_hash": ["a" * 64],
            "trade_count": [0.0],
            "order_count": [0.0],
        }
    )

    diagnostics = case_diagnostic_statistics(table)

    assert diagnostics.columns.tolist() == CASE_DIAGNOSTIC_COLUMNS.tolist()
    first = diagnostics.iloc[0]
    assert first["sweep_rank"] == 1
    assert first["case_name"] == "case-001"
    assert first["case_config_hash"] == "a" * 64
    assert diagnostics.set_index("check").loc["交易样本", "status"] == "失败"
