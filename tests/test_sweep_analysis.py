from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trending_winning.backtest.sweep_analysis import (
    parameter_summary_table,
    pareto_dominance_matrix,
    pareto_front_ranks,
    pareto_sweep_table,
    rank_sweep_table,
)


def test_sweep_analysis_ranks_cases_and_pareto_fronts_without_experiment_config() -> None:
    table = pd.DataFrame(
        {
            "case_name": ["case-a", "case-b", "case-c", "case-d"],
            "case_config_hash": ["a" * 64, "b" * 64, "c" * 64, "d" * 64],
            "total_return": [0.10, 0.12, 0.08, 0.06],
            "max_drawdown": [-0.05, -0.10, -0.06, -0.12],
            "ulcer_index": [0.03, 0.08, 0.04, 0.09],
            "monthly_worst_return": [-0.02, -0.03, -0.05, -0.08],
            "monthly_return_std": [0.03, 0.04, 0.05, 0.06],
            "trade_count": [10, 12, 8, 4],
        }
    )

    ranked = rank_sweep_table(table)
    pareto = pareto_sweep_table(ranked)

    assert ranked.columns[:4].tolist() == ["sweep_rank", "pareto_rank", "is_pareto_efficient", "case_config_hash"]
    assert ranked["case_name"].tolist() == ["case-b", "case-a", "case-c", "case-d"]
    assert ranked.set_index("case_name").loc["case-c", "pareto_rank"] == 2
    assert pareto["pareto_rank"].eq(1).all()


def test_sweep_analysis_adds_risk_adjusted_score_without_replacing_sweep_rank() -> None:
    table = pd.DataFrame(
        {
            "case_name": ["fragile", "steady"],
            "case_config_hash": ["f" * 64, "s" * 64],
            "total_return": [0.20, 0.10],
            "max_drawdown": [-0.20, -0.03],
            "ulcer_index": [0.12, 0.01],
            "monthly_worst_return": [-0.15, -0.01],
            "monthly_return_std": [0.08, 0.01],
            "return_per_exposure_bar": [0.004, 0.003],
            "trade_count": [5, 20],
            "diagnostic_failed_count": [2, 0],
            "diagnostic_attention_count": [1, 0],
            "diagnostic_max_severity": [2, 0],
        }
    )

    ranked = rank_sweep_table(table)
    by_name = ranked.set_index("case_name")

    assert ranked["case_name"].tolist()[0] == "fragile"
    assert by_name.loc["steady", "risk_adjusted_rank"] == 1
    assert by_name.loc["steady", "risk_adjusted_score"] > by_name.loc["fragile", "risk_adjusted_score"]
    assert ranked.columns[:6].tolist() == [
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_config_hash",
        "risk_adjusted_rank",
        "risk_adjusted_score",
    ]


def test_sweep_analysis_uses_single_batch_dominance_matrix_for_fronts() -> None:
    calls = 0
    original = pareto_dominance_matrix

    def spy(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(values)

    table = pd.DataFrame(
        {
            "case_name": ["case-a", "case-b", "case-c", "case-d"],
            "total_return": [0.10, 0.12, 0.08, 0.06],
            "max_drawdown": [-0.05, -0.10, -0.06, -0.12],
            "ulcer_index": [0.03, 0.08, 0.04, 0.09],
            "trade_count": [10, 12, 8, 4],
        }
    )

    assert pareto_front_ranks(table, dominance_matrix_fn=spy) == [1, 1, 2, 3]
    assert calls == 1


def test_sweep_analysis_handles_empty_table_with_risk_score_columns() -> None:
    ranked = rank_sweep_table(pd.DataFrame())

    assert ranked.columns[:5].tolist() == [
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "risk_adjusted_rank",
        "risk_adjusted_score",
    ]
    assert ranked.empty


def test_sweep_analysis_parameter_summary_groups_values_and_formats_structured_values() -> None:
    table = rank_sweep_table(
        pd.DataFrame(
            {
                "case_name": ["a", "b", "c"],
                "case_config_hash": ["a" * 64, "b" * 64, "c" * 64],
                "risk_reward": [2.0, 2.0, 1.5],
                "strategy_priority": [{"trend": 1}, {"trend": 1}, {"range": 2}],
                "total_return": [0.10, -0.02, 0.04],
                "max_drawdown": [-0.04, -0.03, -0.02],
                "monthly_worst_return": [-0.02, -0.04, -0.03],
                "monthly_return_std": [0.01, 0.03, 0.02],
                "trade_count": [5, 3, 4],
                "breakeven_win_rate": [0.35, 0.5, 0.4],
                "win_rate_edge": [0.15, -0.1, 0.05],
                "take_profit_exit_rate": [0.6, 0.2, 0.5],
            }
        )
    )

    summary = parameter_summary_table(table, {"risk_reward": [2.0, 1.5], "strategy_priority": []})
    risk_reward = summary.loc[summary["parameter"].eq("risk_reward") & summary["value"].eq("2.0")].iloc[0]

    assert risk_reward["case_count"] == 2
    assert risk_reward["positive_return_case_count"] == 1.0
    assert risk_reward["positive_return_rate"] == pytest.approx(0.5)
    assert risk_reward["avg_breakeven_win_rate"] == pytest.approx(0.425)
    assert risk_reward["avg_win_rate_edge"] == pytest.approx(0.025)
    assert risk_reward["avg_take_profit_exit_rate"] == pytest.approx(0.4)
    assert "avg_risk_adjusted_score" in summary.columns
    assert '{"trend":1}' in set(summary.loc[summary["parameter"].eq("strategy_priority"), "value"])
