from __future__ import annotations

import pandas as pd

from trending_winning.backtest.sweep_summary import sweep_summary_statistics


def test_sweep_summary_statistics_aggregates_best_case_cache_and_case_tables() -> None:
    table = pd.DataFrame(
        {
            "sweep_rank": [1, 2, 3],
            "pareto_rank": [1, 1, 2],
            "is_pareto_efficient": [True, True, False],
            "case_name": ["best", "second", "third"],
            "case_config_hash": ["a" * 64, "b" * 64, "c" * 64],
            "risk_adjusted_rank": [2, 1, 3],
            "risk_adjusted_score": [58.0, 92.5, 11.5],
            "total_return": [0.12, 0.08, -0.01],
            "max_drawdown": [-0.04, -0.03, -0.02],
            "monthly_worst_return": [-0.02, -0.03, -0.04],
            "order_cache_status": ["miss", "hit", "hit"],
            "candidate_cache_status": ["miss", "miss", "hit"],
            "generated_order_count": [4, 0, 0],
            "candidate_count": [3, 3, 3],
            "candidate_rejection_count": [1, 0, 2],
            "data_inventory_signature": ["sig", "sig", "sig"],
        }
    )
    strategy_stats = pd.DataFrame({"strategy_name": ["trend", "range"], "trade_count": [2, 0]})
    setup_order_decisions = pd.DataFrame(
        {
            "status": ["accepted", "rejected"],
            "reason": ["filled", "no_fill"],
            "decision_count": [2, 3],
        }
    )
    case_diagnostics = pd.DataFrame(
        {
            "case_name": ["best", "best", "second"],
            "status": ["失败", "关注", "通过"],
            "check": ["收益质量", "回撤压力", "数据覆盖"],
        }
    )

    summary = sweep_summary_statistics(
        table=table,
        grid={"risk_reward": [1.5, 2.0], "fee_rate": [0.0]},
        elapsed_seconds=1.25,
        input_bar_count=240,
        filtered_limit_open_count=2,
        strategy_stats=strategy_stats,
        detector_stats=pd.DataFrame(),
        setup_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        setup_order_decision_stats=setup_order_decisions,
        setup_strategy_filter_stats=pd.DataFrame(),
        case_diagnostics=case_diagnostics,
    )

    assert summary["case_count"] == 3
    assert summary["grid_case_count"] == 2
    assert summary["grid_value_counts"] == {"risk_reward": 2, "fee_rate": 1}
    assert summary["pareto_case_count"] == 2
    assert summary["best_case_name"] == "best"
    assert summary["best_total_return"] == 0.12
    assert summary["best_risk_adjusted_case_name"] == "second"
    assert summary["best_risk_adjusted_case_config_hash"] == "b" * 64
    assert summary["best_risk_adjusted_sweep_rank"] == 2
    assert summary["best_risk_adjusted_score"] == 92.5
    assert summary["avg_risk_adjusted_score"] == (58.0 + 92.5 + 11.5) / 3
    assert summary["median_risk_adjusted_score"] == 58.0
    assert summary["worst_risk_adjusted_score"] == 11.5
    assert summary["data_inventory_signature"] == "sig"
    assert summary["order_cache_hit_count"] == 2.0
    assert summary["order_cache_hit_rate"] == 2 / 3
    assert summary["candidate_cache_hit_rate"] == 1 / 3
    assert summary["generated_order_count"] == 4.0
    assert summary["candidate_count"] == 9.0
    assert summary["candidate_rejection_count"] == 3.0
    assert summary["case_strategy_row_count"] == 2.0
    assert summary["case_strategy_trade_count"] == 2.0
    assert summary["case_strategy_zero_trade_row_count"] == 1.0
    assert summary["case_setup_order_decision_count"] == 5.0
    assert summary["case_setup_order_rejected_count"] == 3.0
    assert summary["case_setup_order_rejection_rate"] == 0.6
    assert summary["case_diagnostic_failed_count"] == 1.0
    assert summary["case_diagnostic_attention_count"] == 1.0
    assert summary["case_diagnostic_failed_case_count"] == 1.0


def test_sweep_summary_statistics_returns_stable_empty_defaults() -> None:
    summary = sweep_summary_statistics(
        table=pd.DataFrame(),
        grid={},
        elapsed_seconds=0.5,
        input_bar_count=0,
        filtered_limit_open_count=0,
        strategy_stats=pd.DataFrame(),
        detector_stats=pd.DataFrame(),
        setup_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        setup_order_decision_stats=pd.DataFrame(),
        setup_strategy_filter_stats=pd.DataFrame(),
        case_diagnostics=pd.DataFrame(),
    )

    assert summary["case_count"] == 0
    assert summary["grid_case_count"] == 0
    assert summary["best_case_name"] == ""
    assert summary["best_risk_adjusted_case_name"] == ""
    assert summary["avg_risk_adjusted_score"] == 0.0
    assert summary["order_cache_hit_rate"] == 0.0
    assert summary["case_setup_strategy_filter_rejected_count"] == 0.0
    assert summary["case_diagnostic_failed_count"] == 0.0
