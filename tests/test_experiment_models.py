from __future__ import annotations

import sys

import pandas as pd

from trending_winning.backtest.models import BacktestResult


def test_experiment_models_import_without_experiment_runner() -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_models import (
        PortfolioBenchmarkReport,
        PortfolioExperimentConfig,
        PortfolioExperimentResult,
        PortfolioSweepResult,
        SingleStrategyExperimentConfig,
        SingleStrategyExperimentResult,
        SingleStrategySweepResult,
    )

    portfolio_config = PortfolioExperimentConfig(
        name="portfolio",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="5m",
        start="2026-05-01",
        end="2026-05-31",
    )
    single_config = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="5m",
        start="2026-05-01",
        end="2026-05-31",
        detector="trend",
    )
    backtest = BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={})

    portfolio_result = PortfolioExperimentResult(
        config=portfolio_config,
        backtest=backtest,
        input_bar_count=10,
        filtered_limit_open_count=1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
        elapsed_seconds=0.1,
    )
    single_result = SingleStrategyExperimentResult(
        config=single_config,
        backtest=backtest,
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )
    sweep = PortfolioSweepResult(
        config=portfolio_config,
        grid={"risk_reward": [2.0]},
        table=pd.DataFrame(),
        data_coverage=pd.DataFrame(),
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
    )
    single_sweep = SingleStrategySweepResult(
        config=single_config,
        grid={"risk_reward": [2.0]},
        table=pd.DataFrame(),
        data_coverage=pd.DataFrame(),
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
    )
    report = PortfolioBenchmarkReport(
        experiment_name="portfolio",
        bar_count=10,
        trade_count=2,
        equity_points=3,
        elapsed_seconds=0.5,
        bars_per_second=20.0,
        trades_per_second=4.0,
    )

    assert portfolio_result.config is portfolio_config
    assert single_result.config is single_config
    assert sweep.grid == {"risk_reward": [2.0]}
    assert single_sweep.grid == {"risk_reward": [2.0]}
    assert report.bars_per_second == 20.0
    assert "trending_winning.backtest.experiment" not in sys.modules


def test_experiment_runner_reexports_models_for_compatibility() -> None:
    from trending_winning.backtest import experiment
    from trending_winning.backtest.experiment_models import (
        PortfolioBenchmarkReport,
        PortfolioExperimentConfig,
        PortfolioExperimentResult,
        PortfolioSweepResult,
        SingleStrategyExperimentConfig,
        SingleStrategyExperimentResult,
        SingleStrategySweepResult,
    )

    assert experiment.PortfolioExperimentConfig is PortfolioExperimentConfig
    assert experiment.SingleStrategyExperimentConfig is SingleStrategyExperimentConfig
    assert experiment.PortfolioExperimentResult is PortfolioExperimentResult
    assert experiment.SingleStrategyExperimentResult is SingleStrategyExperimentResult
    assert experiment.PortfolioSweepResult is PortfolioSweepResult
    assert experiment.SingleStrategySweepResult is SingleStrategySweepResult
    assert experiment.PortfolioBenchmarkReport is PortfolioBenchmarkReport
