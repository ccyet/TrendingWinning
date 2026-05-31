from __future__ import annotations

import sys

import pandas as pd

from trending_winning.backtest.experiment_models import SingleStrategyExperimentConfig


def test_experiment_case_stats_imports_without_runner_and_ranks_decisions() -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_case_stats import (
        SETUP_ORDER_DECISION_FIELDS,
        case_decision_statistics,
        concat_case_decision_statistics,
        ranked_case_decision_statistics,
    )

    decisions = pd.DataFrame(
        {
            "detector_name": ["trend", "trend"],
            "event_type": ["H2", "H2"],
            "side": ["long", "long"],
            "status": ["accepted", "rejected"],
            "reason": ["", "no_fill"],
            "actual_risk_pct": [0.02, 0.03],
            "actual_chase_pct": [0.01, 0.02],
            "actual_reward_to_risk": [2.0, 1.5],
        }
    )

    case_stats = case_decision_statistics(
        decisions,
        case_name="case-001",
        case_config_hash="hash-001",
        group_fields=SETUP_ORDER_DECISION_FIELDS,
    )
    ranked = ranked_case_decision_statistics(
        concat_case_decision_statistics([case_stats], group_fields=SETUP_ORDER_DECISION_FIELDS),
        pd.DataFrame(
            {
                "case_config_hash": ["hash-001"],
                "sweep_rank": [1],
                "pareto_rank": [0],
                "is_pareto_efficient": [True],
            }
        ),
        group_fields=SETUP_ORDER_DECISION_FIELDS,
    )

    assert "trending_winning.backtest.experiment" not in sys.modules
    assert ranked["sweep_rank"].tolist() == [1, 1]
    assert ranked["case_name"].tolist() == ["case-001", "case-001"]
    assert ranked["decision_count"].tolist() == [1, 1]
    assert set(ranked["reason"]) == {"", "no_fill"}


def test_symbol_grouped_trade_statistics_keeps_empty_symbol_rows_with_names() -> None:
    from trending_winning.backtest.experiment_case_stats import symbol_grouped_trade_statistics

    config = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ", "600519.SH"),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    trades = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "return_pct": [5.0],
            "holding_bars": [2],
            "r_multiple": [1.2],
            "mae_pct": [-1.0],
            "mfe_pct": [6.0],
            "mae_r": [-0.2],
            "mfe_r": [1.5],
        }
    )

    stats = symbol_grouped_trade_statistics(
        trades,
        config,
        symbol_name_by_code={"000001.SZ": "平安银行", "600519.SH": "贵州茅台"},
    )

    assert stats["stock_name"].tolist() == ["平安银行", "贵州茅台"]
    assert stats["stock_code"].tolist() == ["000001.SZ", "600519.SH"]
    assert stats["trade_count"].tolist() == [1.0, 0.0]


def test_experiment_runner_reexports_case_stats_for_compatibility() -> None:
    from trending_winning.backtest import experiment
    from trending_winning.backtest.experiment_case_stats import (
        case_decision_statistics,
        ranked_case_decision_statistics,
        symbol_grouped_trade_statistics,
    )

    assert experiment._case_decision_statistics is case_decision_statistics
    assert experiment._ranked_case_decision_statistics is ranked_case_decision_statistics
    assert experiment._symbol_grouped_trade_statistics is symbol_grouped_trade_statistics
