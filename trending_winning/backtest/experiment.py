from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd

from trending_winning.backtest.engine import run_order_backtest_from_normalized, run_single_strategy_backtest_from_normalized
from trending_winning.backtest.experiment_cases import (
    case_config_hash as _case_config_hash,
    effective_sweep_grid as _effective_sweep_grid,
    load_sweep_case_config as load_sweep_case_config,
    sweep_parameter_record as _sweep_parameter_record,
    sweep_variants as _sweep_variants,
)
from trending_winning.backtest.experiment_data import (
    load_experiment_data as _load_experiment_data,
    with_data_management_statistics as _with_data_management_statistics,
)
from trending_winning.backtest.experiment_diagnostics import (
    case_diagnostic_statistics as _case_diagnostic_statistics,
    diagnostic_summary_fields as _diagnostic_summary_fields,
    experiment_diagnostic_report as _experiment_diagnostic_report,
)
from trending_winning.backtest.experiment_case_stats import (
    SETUP_ORDER_DECISION_FIELDS as SETUP_ORDER_DECISION_FIELDS,
    SETUP_STRATEGY_FILTER_FIELDS as SETUP_STRATEGY_FILTER_FIELDS,
    case_decision_statistics as _case_decision_statistics,
    case_detector_statistics as _case_detector_statistics,
    case_setup_statistics as _case_setup_statistics,
    case_strategy_statistics as _case_strategy_statistics,
    case_symbol_statistics as _case_symbol_statistics,
    concat_case_decision_statistics as _concat_case_decision_statistics,
    concat_case_detector_statistics as _concat_case_detector_statistics,
    concat_case_setup_statistics as _concat_case_setup_statistics,
    concat_case_strategy_statistics as _concat_case_strategy_statistics,
    concat_case_symbol_statistics as _concat_case_symbol_statistics,
    configured_detector_names as _configured_detector_names,
    ranked_case_decision_statistics as _ranked_case_decision_statistics,
    ranked_case_detector_statistics as _ranked_case_detector_statistics,
    ranked_case_setup_statistics as _ranked_case_setup_statistics,
    ranked_case_strategy_statistics as _ranked_case_strategy_statistics,
    ranked_case_symbol_statistics as _ranked_case_symbol_statistics,
    symbol_grouped_trade_statistics as _symbol_grouped_trade_statistics,
    symbol_name_map_for_config as _symbol_name_map_for_config,
)
from trending_winning.backtest.experiment_config import (
    active_strategy_suite_cache_key as _active_strategy_suite_cache_key,
    backtest_config as _backtest_config,
    candidate_cache_key as _candidate_cache_key,
    detector_cache_parameters as _detector_cache_parameters,
    higher_timeframe_context as _higher_timeframe_context,
    order_cache_key as _order_cache_key,
    portfolio_config as _portfolio_config,
    strategy_suite_config as _strategy_suite_config,
    wrap_higher_timeframe_strategies as _wrap_higher_timeframe_strategies_impl,
    wrap_terminal_false_breakout_strategies as _wrap_terminal_false_breakout_strategies,
)
from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport as PortfolioBenchmarkReport,
    PortfolioExperimentConfig as PortfolioExperimentConfig,
    PortfolioExperimentResult as PortfolioExperimentResult,
    PortfolioSweepResult as PortfolioSweepResult,
    SingleStrategyExperimentConfig as SingleStrategyExperimentConfig,
    SingleStrategyExperimentResult as SingleStrategyExperimentResult,
    SingleStrategySweepResult as SingleStrategySweepResult,
)
from trending_winning.backtest.models import BacktestResult
from trending_winning.backtest.periods import compute_period_return_statistics, compute_period_returns
from trending_winning.backtest.portfolio import (
    prepare_portfolio_candidates_from_normalized,
    run_portfolio_candidate_backtest_from_normalized,
    run_portfolio_backtest_from_normalized,
)
from trending_winning.backtest.portfolio_models import PortfolioCandidateSet
from trending_winning.backtest.reporting import (
    detector_trade_statistics as _detector_trade_statistics,
    grouped_trade_statistics as _grouped_trade_statistics,
    signal_lifecycle_statistics as _signal_lifecycle_statistics,
    setup_trade_statistics as _setup_trade_statistics,
    strategy_names_for_statistics as _strategy_names_for_statistics,
    strategy_trade_statistics as _strategy_trade_statistics,
    trade_path_distribution_statistics as _trade_path_distribution_statistics,
    trade_dated_equity_curve,
)
from trending_winning.backtest.stats import (
    compute_decision_reason_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.backtest.sweep_analysis import (
    pareto_dominance_matrix as _build_pareto_dominance_matrix,
    pareto_front_ranks as _build_pareto_front_ranks,
    pareto_score_table as _build_pareto_score_table,
    rank_sweep_table,
)
from trending_winning.data.repository import MarketDataRepository
from trending_winning.data.summary import summarize_data_management
from trending_winning.backtest.experiment_output import (
    save_portfolio_benchmark as save_portfolio_benchmark,
    save_portfolio_experiment as save_portfolio_experiment,
    save_portfolio_sweep as save_portfolio_sweep,
    save_single_strategy_experiment as save_single_strategy_experiment,
    save_single_strategy_sweep as save_single_strategy_sweep,
)
from trending_winning.strategies.runtime import execute_strategy, execute_strategies
from trending_winning.strategies.suite import create_default_strategy_suite, create_strategy_for_detector

__all__ = [
    "_active_strategy_suite_cache_key",
    "_backtest_config",
    "_candidate_cache_key",
    "_detector_cache_parameters",
    "_higher_timeframe_context",
    "_order_cache_key",
    "_portfolio_config",
    "_strategy_suite_config",
    "_wrap_higher_timeframe_strategies",
]


def _wrap_higher_timeframe_strategies(
    strategies: Sequence[object],
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    higher_bars: pd.DataFrame,
) -> list[object]:
    """兼容旧 monkeypatch 入口；实际大周期包装逻辑在 experiment_config。"""
    return _wrap_higher_timeframe_strategies_impl(
        strategies,
        config,
        higher_bars,
        context_fn=_higher_timeframe_context,
    )


def run_single_strategy_experiment(
    config: SingleStrategyExperimentConfig,
    *,
    save: bool = False,
) -> SingleStrategyExperimentResult:
    start_time = perf_counter()
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)
    strategy = _wrap_higher_timeframe_strategies(
        _wrap_terminal_false_breakout_strategies(
            [create_strategy_for_detector(config.detector, _strategy_suite_config(config))],
            config,
            data.bars,
        ),
        config,
        data.higher_bars,
    )[0]
    backtest = run_single_strategy_backtest_from_normalized(
        data.bars,
        strategy,
        _backtest_config(config),
        timeframe=config.timeframe,
    )
    backtest = _with_data_management_statistics(backtest, data, min_coverage_ratio=config.min_coverage_ratio)
    monthly_returns = compute_period_returns(trade_dated_equity_curve(backtest.equity_curve, backtest.trades), freq="M")
    backtest = _with_period_return_statistics(backtest, monthly_returns)
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=backtest,
        bars=data.bars,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
        data_coverage=data.data_audit,
        data_gap_episodes=data.data_gap_episodes,
        data_inventory=data.data_inventory,
        limit_filter_audit=data.limit_filter_audit,
        strategy_stats=_strategy_trade_statistics(
            backtest.trades,
            (strategy,),
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        symbol_stats=_symbol_grouped_trade_statistics(backtest.trades, config),
        side_stats=_grouped_trade_statistics(backtest.trades, by="side"),
        exit_reason_stats=_grouped_trade_statistics(backtest.trades, by="exit_reason"),
        signal_lifecycle_stats=_signal_lifecycle_statistics(backtest.trades),
        detector_stats=_detector_trade_statistics(
            backtest.trades,
            _configured_detector_names(config),
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        setup_stats=_setup_trade_statistics(
            backtest.trades,
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        event_type_stats=_grouped_trade_statistics(backtest.trades, by="event_type"),
        trade_path_distribution_stats=_trade_path_distribution_statistics(backtest.trades),
        diagnostic_report=_experiment_diagnostic_report(backtest.stats, data_coverage=data.data_audit),
        order_decision_stats=compute_decision_reason_statistics(backtest.order_decisions),
        strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("strategy_name", "filter_name", "context_timeframe"),
        ),
        setup_order_decision_stats=compute_decision_reason_statistics(
            backtest.order_decisions,
            group_fields=("detector_name", "event_type", "side"),
        ),
        setup_strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("detector_name", "event_type", "side", "filter_name", "context_timeframe"),
        ),
        monthly_returns=monthly_returns,
    )
    if save:
        save_single_strategy_experiment(result)
    return result


def run_portfolio_experiment(config: PortfolioExperimentConfig, *, save: bool = False) -> PortfolioExperimentResult:
    start_time = perf_counter()
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)
    strategies = _wrap_higher_timeframe_strategies(
        _wrap_terminal_false_breakout_strategies(
            create_default_strategy_suite(_strategy_suite_config(config)),
            config,
            data.bars,
        ),
        config,
        data.higher_bars,
    )
    backtest = run_portfolio_backtest_from_normalized(
        data.bars,
        strategies,
        _backtest_config(config),
        _portfolio_config(config),
        timeframe=config.timeframe,
    )
    backtest = _with_data_management_statistics(backtest, data, min_coverage_ratio=config.min_coverage_ratio)
    monthly_returns = compute_period_returns(backtest.equity_curve, freq="M")
    backtest = _with_period_return_statistics(backtest, monthly_returns)
    result = PortfolioExperimentResult(
        config=config,
        backtest=backtest,
        bars=data.bars,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        data_coverage=data.data_audit,
        data_gap_episodes=data.data_gap_episodes,
        data_inventory=data.data_inventory,
        limit_filter_audit=data.limit_filter_audit,
        strategy_stats=_strategy_trade_statistics(
            backtest.trades,
            strategies,
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        symbol_stats=_symbol_grouped_trade_statistics(backtest.trades, config),
        side_stats=_grouped_trade_statistics(backtest.trades, by="side"),
        exit_reason_stats=_grouped_trade_statistics(backtest.trades, by="exit_reason"),
        signal_lifecycle_stats=_signal_lifecycle_statistics(backtest.trades),
        detector_stats=_detector_trade_statistics(
            backtest.trades,
            _configured_detector_names(config),
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        setup_stats=_setup_trade_statistics(
            backtest.trades,
            backtest.order_decisions,
            backtest.strategy_filter_decisions,
        ),
        event_type_stats=_grouped_trade_statistics(backtest.trades, by="event_type"),
        trade_path_distribution_stats=_trade_path_distribution_statistics(backtest.trades),
        diagnostic_report=_experiment_diagnostic_report(backtest.stats, data_coverage=data.data_audit),
        order_decision_stats=compute_decision_reason_statistics(backtest.order_decisions),
        strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("strategy_name", "filter_name", "context_timeframe"),
        ),
        setup_order_decision_stats=compute_decision_reason_statistics(
            backtest.order_decisions,
            group_fields=("detector_name", "event_type", "side"),
        ),
        setup_strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("detector_name", "event_type", "side", "filter_name", "context_timeframe"),
        ),
        monthly_returns=monthly_returns,
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
    )
    if save:
        save_portfolio_experiment(result)
    return result


def run_portfolio_parameter_sweep(
    config: PortfolioExperimentConfig,
    *,
    grid: Mapping[str, Sequence[object]],
    save: bool = False,
) -> PortfolioSweepResult:
    start_time = perf_counter()
    variants = _sweep_variants(config, grid)
    effective_grid = _effective_sweep_grid(config, grid)
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)

    rows: list[dict[str, object]] = []
    strategy_frames: list[pd.DataFrame] = []
    detector_frames: list[pd.DataFrame] = []
    setup_frames: list[pd.DataFrame] = []
    symbol_frames: list[pd.DataFrame] = []
    setup_order_decision_frames: list[pd.DataFrame] = []
    setup_strategy_filter_frames: list[pd.DataFrame] = []
    symbol_name_by_code = _symbol_name_map_for_config(config)
    data_stats = summarize_data_management(
        data.data_audit,
        data.limit_filter_audit,
        filtered_limit_open_count=len(data.filtered_limit_open_days),
        data_inventory=data.data_inventory,
        min_coverage_ratio=config.min_coverage_ratio,
    )
    orders_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
    strategy_names_by_config: dict[tuple[object, ...], tuple[str, ...]] = {}
    filter_decisions_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
    candidates_by_execution: dict[tuple[tuple[object, ...], tuple[object, ...]], PortfolioCandidateSet] = {}
    for case_index, variant in enumerate(variants, start=1):
        case_start = perf_counter()
        suite_config = _strategy_suite_config(variant)
        order_key = _order_cache_key(variant, suite_config)
        orders = orders_by_config.get(order_key)
        order_cache_status = "hit"
        if orders is None:
            order_cache_status = "miss"
            strategies = _wrap_higher_timeframe_strategies(
                _wrap_terminal_false_breakout_strategies(
                    create_default_strategy_suite(suite_config),
                    variant,
                    data.bars,
                ),
                variant,
                data.higher_bars,
            )
            strategy_runs = execute_strategies(strategies, data.bars, timeframe=variant.timeframe)
            orders = strategy_runs.orders
            orders_by_config[order_key] = orders
            strategy_names_by_config[order_key] = _strategy_names_for_statistics(strategies)
            filter_decisions_by_config[order_key] = strategy_runs.filter_decisions
        backtest_config = _backtest_config(variant)
        candidate_key = (order_key, _candidate_cache_key(variant))
        candidate_set = candidates_by_execution.get(candidate_key)
        candidate_cache_status = "hit"
        if candidate_set is None:
            candidate_cache_status = "miss"
            candidate_set = prepare_portfolio_candidates_from_normalized(data.bars, orders, backtest_config)
            candidates_by_execution[candidate_key] = candidate_set
        filter_decisions = filter_decisions_by_config.get(order_key, pd.DataFrame())
        backtest = run_portfolio_candidate_backtest_from_normalized(
            data.bars,
            candidate_set,
            backtest_config,
            _portfolio_config(variant),
        )
        case_elapsed = max(perf_counter() - case_start, 1e-12)
        case_name = f"{config.name}-{case_index:03d}"
        case_hash = _case_config_hash(variant)
        row: dict[str, object] = {
            "case_name": case_name,
            "case_config_hash": case_hash,
            "bar_count": int(len(data.bars)),
            "trade_count": int(len(backtest.trades)),
            "equity_points": int(len(backtest.equity_curve)),
            "elapsed_seconds": float(case_elapsed),
            "bars_per_second": float(len(data.bars) / case_elapsed),
            "order_cache_status": order_cache_status,
            "candidate_cache_status": candidate_cache_status,
            "generated_order_count": int(len(orders)),
            "candidate_count": int(len(candidate_set.candidates)),
            "candidate_rejection_count": int(len(candidate_set.rejections)),
        }
        row.update(_sweep_parameter_record(config, variant, effective_grid.keys()))
        row.update(data_stats)
        row.update(summarize_order_decisions(backtest.order_decisions))
        row.update(backtest.stats)
        row.update(_monthly_period_statistics(backtest, use_trade_dates=False))
        row.update(summarize_strategy_filter_decisions(filter_decisions))
        row.update(_diagnostic_summary_fields(row))
        rows.append(row)
        strategy_frames.append(
            _case_strategy_statistics(
                backtest.trades,
                strategy_names_by_config.get(order_key, ()),
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        detector_frames.append(
            _case_detector_statistics(
                backtest.trades,
                variant,
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        setup_frames.append(
            _case_setup_statistics(
                backtest.trades,
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        symbol_frames.append(
            _case_symbol_statistics(
                backtest.trades,
                variant,
                symbol_name_by_code=symbol_name_by_code,
                case_name=case_name,
                case_config_hash=case_hash,
            )
        )
        setup_order_decision_frames.append(
            _case_decision_statistics(
                backtest.order_decisions,
                case_name=case_name,
                case_config_hash=case_hash,
                group_fields=SETUP_ORDER_DECISION_FIELDS,
            )
        )
        setup_strategy_filter_frames.append(
            _case_decision_statistics(
                filter_decisions,
                case_name=case_name,
                case_config_hash=case_hash,
                group_fields=SETUP_STRATEGY_FILTER_FIELDS,
            )
        )

    table = _rank_sweep_table(pd.DataFrame(rows))
    strategy_stats = _ranked_case_strategy_statistics(_concat_case_strategy_statistics(strategy_frames), table)
    detector_stats = _ranked_case_detector_statistics(_concat_case_detector_statistics(detector_frames), table)
    setup_stats = _ranked_case_setup_statistics(_concat_case_setup_statistics(setup_frames), table)
    symbol_stats = _ranked_case_symbol_statistics(_concat_case_symbol_statistics(symbol_frames), table)
    setup_order_decision_stats = _ranked_case_decision_statistics(
        _concat_case_decision_statistics(setup_order_decision_frames, group_fields=SETUP_ORDER_DECISION_FIELDS),
        table,
        group_fields=SETUP_ORDER_DECISION_FIELDS,
    )
    setup_strategy_filter_stats = _ranked_case_decision_statistics(
        _concat_case_decision_statistics(setup_strategy_filter_frames, group_fields=SETUP_STRATEGY_FILTER_FIELDS),
        table,
        group_fields=SETUP_STRATEGY_FILTER_FIELDS,
    )
    case_diagnostics = _case_diagnostic_statistics(table)
    result = PortfolioSweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
        data_gap_episodes=data.data_gap_episodes,
        data_inventory=data.data_inventory,
        limit_filter_audit=data.limit_filter_audit,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
        strategy_stats=strategy_stats,
        detector_stats=detector_stats,
        setup_stats=setup_stats,
        symbol_stats=symbol_stats,
        setup_order_decision_stats=setup_order_decision_stats,
        setup_strategy_filter_stats=setup_strategy_filter_stats,
        case_diagnostics=case_diagnostics,
    )
    if save:
        save_portfolio_sweep(result)
    return result


def run_single_strategy_parameter_sweep(
    config: SingleStrategyExperimentConfig,
    *,
    grid: Mapping[str, Sequence[object]],
    save: bool = False,
) -> SingleStrategySweepResult:
    """单策略参数遍历；一次加载数据，按订单参数缓存信号订单。"""
    start_time = perf_counter()
    variants = _sweep_variants(config, grid)
    effective_grid = _effective_sweep_grid(config, grid)
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)

    rows: list[dict[str, object]] = []
    strategy_frames: list[pd.DataFrame] = []
    detector_frames: list[pd.DataFrame] = []
    setup_frames: list[pd.DataFrame] = []
    symbol_frames: list[pd.DataFrame] = []
    setup_order_decision_frames: list[pd.DataFrame] = []
    setup_strategy_filter_frames: list[pd.DataFrame] = []
    symbol_name_by_code = _symbol_name_map_for_config(config)
    data_stats = summarize_data_management(
        data.data_audit,
        data.limit_filter_audit,
        filtered_limit_open_count=len(data.filtered_limit_open_days),
        data_inventory=data.data_inventory,
        min_coverage_ratio=config.min_coverage_ratio,
    )
    orders_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
    strategy_names_by_config: dict[tuple[object, ...], tuple[str, ...]] = {}
    filter_decisions_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
    for case_index, variant in enumerate(variants, start=1):
        case_start = perf_counter()
        suite_config = _strategy_suite_config(variant)
        order_key = _order_cache_key(variant, suite_config)
        orders = orders_by_config.get(order_key)
        order_cache_status = "hit"
        if orders is None:
            order_cache_status = "miss"
            strategy = _wrap_higher_timeframe_strategies(
                _wrap_terminal_false_breakout_strategies(
                    [create_strategy_for_detector(variant.detector, suite_config)],
                    variant,
                    data.bars,
                ),
                variant,
                data.higher_bars,
            )[0]
            strategy_run = execute_strategy(strategy, data.bars, timeframe=variant.timeframe)
            orders = strategy_run.orders
            orders_by_config[order_key] = orders
            strategy_names_by_config[order_key] = _strategy_names_for_statistics((strategy,))
            filter_decisions_by_config[order_key] = strategy_run.filter_decisions
        filter_decisions = filter_decisions_by_config.get(order_key, pd.DataFrame())
        backtest = _with_strategy_filter_decisions(
            run_order_backtest_from_normalized(data.bars, orders, _backtest_config(variant)),
            filter_decisions,
        )
        case_elapsed = max(perf_counter() - case_start, 1e-12)
        case_name = f"{config.name}-{case_index:03d}"
        case_hash = _case_config_hash(variant)
        row: dict[str, object] = {
            "case_name": case_name,
            "case_config_hash": case_hash,
            "bar_count": int(len(data.bars)),
            "trade_count": int(len(backtest.trades)),
            "equity_points": int(len(backtest.equity_curve)),
            "elapsed_seconds": float(case_elapsed),
            "bars_per_second": float(len(data.bars) / case_elapsed),
            "order_cache_status": order_cache_status,
            "generated_order_count": int(len(orders)),
        }
        row.update(_sweep_parameter_record(config, variant, effective_grid.keys()))
        row.update(data_stats)
        row.update(summarize_order_decisions(backtest.order_decisions))
        row.update(backtest.stats)
        row.update(_monthly_period_statistics(backtest, use_trade_dates=True))
        row.update(summarize_strategy_filter_decisions(filter_decisions))
        row.update(_diagnostic_summary_fields(row))
        rows.append(row)
        strategy_frames.append(
            _case_strategy_statistics(
                backtest.trades,
                strategy_names_by_config.get(order_key, ()),
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        detector_frames.append(
            _case_detector_statistics(
                backtest.trades,
                variant,
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        setup_frames.append(
            _case_setup_statistics(
                backtest.trades,
                case_name=case_name,
                case_config_hash=case_hash,
                order_decisions=backtest.order_decisions,
                filter_decisions=filter_decisions,
            )
        )
        symbol_frames.append(
            _case_symbol_statistics(
                backtest.trades,
                variant,
                symbol_name_by_code=symbol_name_by_code,
                case_name=case_name,
                case_config_hash=case_hash,
            )
        )
        setup_order_decision_frames.append(
            _case_decision_statistics(
                backtest.order_decisions,
                case_name=case_name,
                case_config_hash=case_hash,
                group_fields=SETUP_ORDER_DECISION_FIELDS,
            )
        )
        setup_strategy_filter_frames.append(
            _case_decision_statistics(
                filter_decisions,
                case_name=case_name,
                case_config_hash=case_hash,
                group_fields=SETUP_STRATEGY_FILTER_FIELDS,
            )
        )

    table = _rank_sweep_table(pd.DataFrame(rows))
    strategy_stats = _ranked_case_strategy_statistics(_concat_case_strategy_statistics(strategy_frames), table)
    detector_stats = _ranked_case_detector_statistics(_concat_case_detector_statistics(detector_frames), table)
    setup_stats = _ranked_case_setup_statistics(_concat_case_setup_statistics(setup_frames), table)
    symbol_stats = _ranked_case_symbol_statistics(_concat_case_symbol_statistics(symbol_frames), table)
    setup_order_decision_stats = _ranked_case_decision_statistics(
        _concat_case_decision_statistics(setup_order_decision_frames, group_fields=SETUP_ORDER_DECISION_FIELDS),
        table,
        group_fields=SETUP_ORDER_DECISION_FIELDS,
    )
    setup_strategy_filter_stats = _ranked_case_decision_statistics(
        _concat_case_decision_statistics(setup_strategy_filter_frames, group_fields=SETUP_STRATEGY_FILTER_FIELDS),
        table,
        group_fields=SETUP_STRATEGY_FILTER_FIELDS,
    )
    case_diagnostics = _case_diagnostic_statistics(table)
    result = SingleStrategySweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
        data_gap_episodes=data.data_gap_episodes,
        data_inventory=data.data_inventory,
        limit_filter_audit=data.limit_filter_audit,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
        strategy_stats=strategy_stats,
        detector_stats=detector_stats,
        setup_stats=setup_stats,
        symbol_stats=symbol_stats,
        setup_order_decision_stats=setup_order_decision_stats,
        setup_strategy_filter_stats=setup_strategy_filter_stats,
        case_diagnostics=case_diagnostics,
    )
    if save:
        save_single_strategy_sweep(result)
    return result


def _with_strategy_filter_decisions(result: BacktestResult, filter_decisions: pd.DataFrame) -> BacktestResult:
    stats = dict(result.stats)
    stats.update(summarize_strategy_filter_decisions(filter_decisions))
    return BacktestResult(
        trades=result.trades,
        equity_curve=result.equity_curve,
        stats=stats,
        order_decisions=result.order_decisions,
        strategy_filter_decisions=filter_decisions,
    )


def _with_period_return_statistics(result: BacktestResult, period_returns: pd.DataFrame) -> BacktestResult:
    """运行态结果同步携带周期稳定性摘要，避免 Web/CLI 和落盘文件口径分裂。"""
    stats = dict(result.stats)
    stats.update(compute_period_return_statistics(period_returns, prefix="monthly"))
    return BacktestResult(
        trades=result.trades,
        equity_curve=result.equity_curve,
        stats=stats,
        order_decisions=result.order_decisions,
        strategy_filter_decisions=result.strategy_filter_decisions,
    )


def _monthly_period_statistics(result: BacktestResult, *, use_trade_dates: bool) -> dict[str, object]:
    """给参数遍历行计算月度稳定性；组合用盯市净值，单策略用成交时间轴。"""
    equity_curve = trade_dated_equity_curve(result.equity_curve, result.trades) if use_trade_dates else result.equity_curve
    monthly_returns = compute_period_returns(equity_curve, freq="M")
    return compute_period_return_statistics(monthly_returns, prefix="monthly")


def benchmark_portfolio_experiment(config: PortfolioExperimentConfig, *, save: bool = False) -> PortfolioBenchmarkReport:
    result = run_portfolio_experiment(config, save=save)
    report = build_portfolio_benchmark_report(result)
    if save:
        save_portfolio_benchmark(config, report)
    return report


def build_portfolio_benchmark_report(result: PortfolioExperimentResult) -> PortfolioBenchmarkReport:
    """从已完成的组合实验生成性能报告，避免 benchmark 模式重复跑回测。"""
    elapsed = max(float(result.elapsed_seconds), 1e-12)
    bar_count = int(result.input_bar_count)
    trade_count = int(len(result.backtest.trades))
    return PortfolioBenchmarkReport(
        experiment_name=result.config.name,
        bar_count=bar_count,
        trade_count=trade_count,
        equity_points=int(len(result.backtest.equity_curve)),
        elapsed_seconds=float(elapsed),
        bars_per_second=float(bar_count / elapsed),
        trades_per_second=float(trade_count / elapsed),
    )


def _rank_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    return rank_sweep_table(table)


def _pareto_front_ranks(table: pd.DataFrame) -> list[int]:
    return _build_pareto_front_ranks(table, dominance_matrix_fn=_pareto_dominance_matrix)


def _pareto_dominance_matrix(values: np.ndarray) -> np.ndarray:
    return _build_pareto_dominance_matrix(values)


def _pareto_score_table(table: pd.DataFrame) -> pd.DataFrame:
    return _build_pareto_score_table(table)
