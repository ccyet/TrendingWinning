from __future__ import annotations

import json
import sys

import pandas as pd

from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport,
    PortfolioExperimentConfig,
    SingleStrategyExperimentConfig,
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
