from __future__ import annotations

import sys

import pandas as pd

from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig


def test_experiment_config_imports_without_runner_and_maps_backtest_portfolio_and_suite_configs() -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_config import backtest_config, portfolio_config, strategy_suite_config

    config = PortfolioExperimentConfig(
        name="portfolio",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend", "range"),
        risk_reward=1.8,
        max_holding_bars=9,
        max_actual_risk_pct=0.08,
        max_chase_pct=0.04,
        side_mode="long_only",
        max_open_positions=3,
        capital_per_trade=0.2,
        risk_per_trade=0.01,
        max_capital_per_trade=0.5,
        short_margin_rate=1.2,
        reserve_cash=0.1,
        allow_same_symbol_overlap=True,
        strategy_priority={"trend_signal_bar": 1},
        strategy_capital_limit={"trend_signal_bar": 0.5},
        sector_capital_limit={"银行": 0.4},
        symbol_sector_map={"000001.SZ": "银行"},
        fee_rate=0.0003,
        slippage_bps=5.0,
        initial_equity=2.0,
        trailing_take_profit_activation_pct=0.05,
        trailing_take_profit_drawdown_pct=0.02,
        trailing_take_profit_ma_period=10,
    )

    bt = backtest_config(config)
    pf = portfolio_config(config)
    suite = strategy_suite_config(config)

    assert "trending_winning.backtest.experiment" not in sys.modules
    assert bt.max_holding_bars == 9
    assert bt.fee_rate == 0.0003
    assert bt.trailing_take_profit_ma_period == 10
    assert pf.max_open_positions == 3
    assert pf.strategy_capital_limit == {"trend_signal_bar": 0.5}
    assert pf.symbol_sector_map == {"000001.SZ": "银行"}
    assert suite.enabled == ("trend", "range")
    assert suite.risk_reward == 1.8
    assert suite.side_mode == "long_only"


def test_experiment_config_order_cache_key_ignores_disabled_detector_parameters_but_includes_context() -> None:
    from trending_winning.backtest.experiment_config import order_cache_key, strategy_suite_config

    base = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        channel_lookback=40,
        higher_timeframe="",
    )
    disabled_changed = SingleStrategyExperimentConfig(
        **{**base.__dict__, "channel_lookback": 99},
    )
    with_context = SingleStrategyExperimentConfig(
        **{**base.__dict__, "higher_timeframe": "60m", "higher_timeframe_max_age_minutes": 90},
    )

    assert order_cache_key(base, strategy_suite_config(base)) == order_cache_key(
        disabled_changed,
        strategy_suite_config(disabled_changed),
    )
    assert order_cache_key(base, strategy_suite_config(base)) != order_cache_key(
        with_context,
        strategy_suite_config(with_context),
    )


def test_experiment_config_wraps_strategies_with_higher_timeframe_gate() -> None:
    from trending_winning.backtest.experiment_config import wrap_higher_timeframe_strategies
    from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy

    class Strategy:
        name = "fixed"

    config = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="15m",
        higher_timeframe="60m",
        higher_timeframe_max_age_minutes=45,
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )

    wrapped = wrap_higher_timeframe_strategies([Strategy()], config, pd.DataFrame())

    assert isinstance(wrapped[0], HigherTimeframeAlignmentStrategy)
    assert wrapped[0].config.context_timeframe == "60m"
    assert wrapped[0].config.max_context_age == pd.Timedelta(minutes=45)


def test_experiment_runner_reexports_config_helpers_for_compatibility() -> None:
    from trending_winning.backtest import experiment
    from trending_winning.backtest.experiment_config import (
        active_strategy_suite_cache_key,
        backtest_config,
        candidate_cache_key,
        detector_cache_parameters,
        higher_timeframe_context,
        order_cache_key,
        portfolio_config,
        strategy_suite_config,
    )

    assert experiment._backtest_config is backtest_config
    assert experiment._portfolio_config is portfolio_config
    assert experiment._candidate_cache_key is candidate_cache_key
    assert experiment._order_cache_key is order_cache_key
    assert experiment._strategy_suite_config is strategy_suite_config
    assert experiment._active_strategy_suite_cache_key is active_strategy_suite_cache_key
    assert experiment._detector_cache_parameters is detector_cache_parameters
    assert experiment._higher_timeframe_context is higher_timeframe_context
