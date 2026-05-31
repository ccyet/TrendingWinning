from __future__ import annotations

import pandas as pd

from trending_winning.backtest.experiment_diagnostics import (
    EXPERIMENT_DIAGNOSTIC_COLUMNS,
    experiment_diagnostic_report,
)


def test_experiment_diagnostic_report_flags_core_risks() -> None:
    stats = {
        "trade_count": 12.0,
        "order_count": 60.0,
        "acceptance_rate": 0.18,
        "strategy_filter_rejection_rate": 0.72,
        "max_drawdown": -0.24,
        "profit_factor": 0.9,
        "monthly_worst_return": -0.11,
        "avg_mae_r": -0.9,
        "max_margin_exposure": 1.2,
        "data_weighted_coverage_ratio": 0.91,
        "data_coverage_below_min_count": 2.0,
    }

    report = experiment_diagnostic_report(stats)

    assert report.columns.tolist() == EXPERIMENT_DIAGNOSTIC_COLUMNS.tolist()
    by_check = report.set_index("check")
    assert by_check.loc["数据覆盖", "status"] == "失败"
    assert by_check.loc["交易样本", "status"] == "关注"
    assert by_check.loc["订单接受率", "status"] == "关注"
    assert by_check.loc["策略过滤", "status"] == "关注"
    assert by_check.loc["回撤压力", "status"] == "关注"
    assert by_check.loc["收益质量", "status"] == "失败"
    assert by_check.loc["路径风险", "status"] == "关注"
    assert by_check.loc["资金暴露", "status"] == "关注"


def test_experiment_diagnostic_report_marks_zero_trade_as_failed() -> None:
    report = experiment_diagnostic_report({"trade_count": 0.0, "order_count": 0.0})

    by_check = report.set_index("check")
    assert by_check.loc["交易样本", "status"] == "失败"
    assert by_check.loc["订单接受率", "status"] == "失败"
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
