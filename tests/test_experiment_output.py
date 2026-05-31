from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from trending_winning.backtest.models import BacktestResult
from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport,
    PortfolioExperimentConfig,
    SingleStrategyExperimentConfig,
    SingleStrategyExperimentResult,
    SingleStrategySweepResult,
)


def test_experiment_output_imports_without_experiment_runner_and_saves_sweep(tmp_path) -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_output import save_single_strategy_sweep

    config = SingleStrategyExperimentConfig(
        name="single-sweep",
        data_root="/data",
        output_dir=str(tmp_path / "runs"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    result = SingleStrategySweepResult(
        config=config,
        grid={"risk_reward": [2.0]},
        table=pd.DataFrame(),
        data_coverage=pd.DataFrame(),
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
    )

    output_dir = save_single_strategy_sweep(result)

    saved_config = json.loads((output_dir / "config.json").read_text())
    saved_cases = [json.loads(line) for line in (output_dir / "case_configs.jsonl").read_text().splitlines()]
    assert "trending_winning.backtest.experiment" not in sys.modules
    assert saved_config["sweep_grid"] == {"risk_reward": [2.0]}
    assert saved_cases[0]["case_name"] == "single-sweep-001"
    assert (output_dir / "parameter_summary.csv").exists()
    assert (output_dir / "symbol_metadata.csv").exists()


def test_experiment_runner_reexports_output_functions_for_compatibility() -> None:
    from trending_winning.backtest import experiment
    from trending_winning.backtest.experiment_output import (
        save_portfolio_benchmark,
        save_portfolio_experiment,
        save_portfolio_sweep,
        save_single_strategy_experiment,
        save_single_strategy_sweep,
    )

    assert experiment.save_single_strategy_experiment is save_single_strategy_experiment
    assert experiment.save_portfolio_experiment is save_portfolio_experiment
    assert experiment.save_portfolio_sweep is save_portfolio_sweep
    assert experiment.save_single_strategy_sweep is save_single_strategy_sweep
    assert experiment.save_portfolio_benchmark is save_portfolio_benchmark


def test_save_portfolio_benchmark_writes_strict_json(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_portfolio_benchmark

    config = PortfolioExperimentConfig(
        name="bench",
        data_root="/data",
        output_dir=str(tmp_path / "bench"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
    )
    report = PortfolioBenchmarkReport(
        experiment_name="bench",
        bar_count=10,
        trade_count=2,
        equity_points=3,
        elapsed_seconds=0.5,
        bars_per_second=20.0,
        trades_per_second=4.0,
    )

    output_dir = save_portfolio_benchmark(config, report)

    assert json.loads((output_dir / "benchmark.json").read_text())["bars_per_second"] == 20.0


def test_save_single_strategy_experiment_writes_drawdown_episodes(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-drawdown",
        data_root="/data",
        output_dir=str(tmp_path / "single-drawdown"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-25", periods=5),
            "net_value": [1.0, 1.2, 1.0, 0.9, 1.21],
            "drawdown_net_value": [1.0, 1.2, 1.0, 0.9, 1.21],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=equity,
            stats={"trade_count": 0.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "drawdown_episodes.csv")
    assert saved.loc[0, "depth"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert saved.loc[0, "start_at"] == "2026-05-26 00:00:00"
    assert saved.loc[0, "trough_at"] == "2026-05-28 00:00:00"


def test_save_single_strategy_experiment_writes_trade_path_distribution(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-path",
        data_root="/data",
        output_dir=str(tmp_path / "single-path"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    trades = pd.DataFrame(
        {
            "return_pct": [3.0, -1.0],
            "holding_bars": [1, 10],
            "r_multiple": [0.8, -1.2],
            "mae_r": [-0.2, -1.1],
            "mfe_r": [1.0, 0.1],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=trades,
            equity_curve=pd.DataFrame(),
            stats={"trade_count": 2.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "trade_path_distribution.csv")
    assert {"dimension", "bucket", "trade_count", "win_rate", "avg_return"}.issubset(saved.columns)
    assert saved.loc[saved["bucket"].eq("9-16K"), "trade_count"].iloc[0] == 1.0


def test_save_single_strategy_experiment_writes_experiment_diagnostics(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-diagnostics",
        data_root="/data",
        output_dir=str(tmp_path / "single-diagnostics"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            stats={"trade_count": 0.0, "order_count": 0.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "experiment_diagnostics.csv")
    assert {"section", "check", "status", "detail"}.issubset(saved.columns)
    assert saved.loc[saved["check"].eq("交易样本"), "status"].iloc[0] == "失败"
