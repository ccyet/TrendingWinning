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
from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport as PortfolioBenchmarkReport,
    PortfolioExperimentConfig as PortfolioExperimentConfig,
    PortfolioExperimentResult as PortfolioExperimentResult,
    PortfolioSweepResult as PortfolioSweepResult,
    SingleStrategyExperimentConfig as SingleStrategyExperimentConfig,
    SingleStrategyExperimentResult as SingleStrategyExperimentResult,
    SingleStrategySweepResult as SingleStrategySweepResult,
)
from trending_winning.backtest.models import BacktestConfig, BacktestResult
from trending_winning.backtest.periods import compute_period_return_statistics, compute_period_returns
from trending_winning.backtest.portfolio import (
    prepare_portfolio_candidates_from_normalized,
    run_portfolio_candidate_backtest_from_normalized,
    run_portfolio_backtest_from_normalized,
)
from trending_winning.backtest.portfolio_models import PortfolioConfig, PortfolioCandidateSet
from trending_winning.backtest.reporting import (
    SETUP_STAT_FIELDS,
    detector_trade_statistics as _detector_trade_statistics,
    grouped_trade_statistics as _grouped_trade_statistics,
    setup_trade_statistics as _setup_trade_statistics,
    strategy_names_for_statistics as _strategy_names_for_statistics,
    strategy_trade_statistics as _strategy_trade_statistics,
    trade_dated_equity_curve,
)
from trending_winning.backtest.stats import (
    STAT_KEYS,
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
from trending_winning.data.schema import unique_symbols
from trending_winning.data.summary import summarize_data_management
from trending_winning.backtest.experiment_output import (
    save_portfolio_benchmark as save_portfolio_benchmark,
    save_portfolio_experiment as save_portfolio_experiment,
    save_portfolio_sweep as save_portfolio_sweep,
    save_single_strategy_experiment as save_single_strategy_experiment,
    save_single_strategy_sweep as save_single_strategy_sweep,
    symbol_metadata_for_config as _symbol_metadata_for_config,
)
from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy, TimeframeAlignmentConfig
from trending_winning.strategies.runtime import execute_strategy, execute_strategies
from trending_winning.strategies.suite import StrategySuiteConfig, create_default_strategy_suite, create_strategy_for_detector

SETUP_ORDER_DECISION_FIELDS = ("detector_name", "event_type", "side")
SETUP_STRATEGY_FILTER_FIELDS = ("detector_name", "event_type", "side", "filter_name", "context_timeframe")
SWEEP_CASE_STRATEGY_COLUMNS = (
    "sweep_rank",
    "pareto_rank",
    "is_pareto_efficient",
    "case_name",
    "case_config_hash",
    "strategy_name",
    *STAT_KEYS,
)
SWEEP_CASE_DETECTOR_COLUMNS = (
    "sweep_rank",
    "pareto_rank",
    "is_pareto_efficient",
    "case_name",
    "case_config_hash",
    "detector_name",
    *STAT_KEYS,
)
SWEEP_CASE_SETUP_COLUMNS = (
    "sweep_rank",
    "pareto_rank",
    "is_pareto_efficient",
    "case_name",
    "case_config_hash",
    *SETUP_STAT_FIELDS,
    *STAT_KEYS,
)
SWEEP_CASE_SYMBOL_COLUMNS = (
    "sweep_rank",
    "pareto_rank",
    "is_pareto_efficient",
    "case_name",
    "case_config_hash",
    "stock_name",
    "stock_code",
    *STAT_KEYS,
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
        [create_strategy_for_detector(config.detector, _strategy_suite_config(config))],
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
        create_default_strategy_suite(_strategy_suite_config(config)),
        config,
        data.higher_bars,
    )
    backtest = run_portfolio_backtest_from_normalized(
        data.bars,
        strategies,
        _backtest_config(config),
        PortfolioConfig(
            max_open_positions=config.max_open_positions,
            capital_per_trade=config.capital_per_trade,
            risk_per_trade=config.risk_per_trade,
            max_capital_per_trade=config.max_capital_per_trade,
            short_margin_rate=config.short_margin_rate,
            reserve_cash=config.reserve_cash,
            allow_same_symbol_overlap=config.allow_same_symbol_overlap,
            strategy_priority=config.strategy_priority,
            strategy_capital_limit=config.strategy_capital_limit,
            sector_capital_limit=config.sector_capital_limit,
            symbol_sector_map=config.symbol_sector_map,
            sector_metadata_key=config.sector_metadata_key,
            default_sector=config.default_sector,
        ),
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
                create_default_strategy_suite(suite_config),
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
            PortfolioConfig(
                max_open_positions=variant.max_open_positions,
                capital_per_trade=variant.capital_per_trade,
                risk_per_trade=variant.risk_per_trade,
                max_capital_per_trade=variant.max_capital_per_trade,
                short_margin_rate=variant.short_margin_rate,
                reserve_cash=variant.reserve_cash,
                allow_same_symbol_overlap=variant.allow_same_symbol_overlap,
                strategy_priority=variant.strategy_priority,
                strategy_capital_limit=variant.strategy_capital_limit,
                sector_capital_limit=variant.sector_capital_limit,
                symbol_sector_map=variant.symbol_sector_map,
                sector_metadata_key=variant.sector_metadata_key,
                default_sector=variant.default_sector,
            ),
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
    result = PortfolioSweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
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
                [create_strategy_for_detector(variant.detector, suite_config)],
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
    result = SingleStrategySweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
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
    )
    if save:
        save_single_strategy_sweep(result)
    return result


def _wrap_higher_timeframe_strategies(
    strategies: Sequence[object],
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    higher_bars: pd.DataFrame,
) -> list[object]:
    higher_timeframe = str(config.higher_timeframe).strip()
    if not higher_timeframe:
        return list(strategies)
    context = _higher_timeframe_context(higher_bars, config)
    max_age = (
        None
        if config.higher_timeframe_max_age_minutes is None
        else pd.Timedelta(minutes=int(config.higher_timeframe_max_age_minutes))
    )
    return [
        HigherTimeframeAlignmentStrategy(
            strategy,
            context,
            TimeframeAlignmentConfig(
                name=f"{strategy.name}_mtf_{higher_timeframe}",
                context_timeframe=higher_timeframe,
                context_column="trend_state",
                max_context_age=max_age,
            ),
        )
        for strategy in strategies
    ]


def _higher_timeframe_context(
    bars: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["date", "stock_code", "trend_state"])
    from trending_winning.detectors.trend import TrendDetectorConfig, attach_trend_state

    trend = attach_trend_state(
        bars,
        TrendDetectorConfig(
            lookback=config.trend_lookback,
            min_trend_score=config.trend_min_score,
            strong_close_pos=config.trend_strong_close_pos,
            min_body_ratio=config.trend_min_body_ratio,
            pullback_lookback=config.trend_pullback_lookback,
            h2_min_pullback_legs=config.trend_h2_min_pullback_legs,
        ),
    )
    if trend.empty:
        return pd.DataFrame(columns=["date", "stock_code", "trend_state"])
    return trend[["date", "stock_code", "trend_state"]].copy()


def _backtest_config(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> BacktestConfig:
    return BacktestConfig(
        max_holding_bars=config.max_holding_bars,
        fee_rate=config.fee_rate,
        slippage_bps=config.slippage_bps,
        initial_equity=config.initial_equity,
        intrabar_exit_policy=config.intrabar_exit_policy,
        trailing_take_profit_activation_pct=config.trailing_take_profit_activation_pct,
        trailing_take_profit_drawdown_pct=config.trailing_take_profit_drawdown_pct,
        trailing_take_profit_ma_period=config.trailing_take_profit_ma_period,
    )


def _candidate_cache_key(config: PortfolioExperimentConfig) -> tuple[object, ...]:
    """候选成交只依赖撮合路径参数；组合分配和初始资金变化可复用。"""
    return (
        int(config.max_holding_bars),
        float(config.fee_rate),
        float(config.slippage_bps),
        str(config.intrabar_exit_policy),
        float(config.trailing_take_profit_activation_pct),
        float(config.trailing_take_profit_drawdown_pct),
        int(config.trailing_take_profit_ma_period),
    )


def _order_cache_key(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    suite_config: StrategySuiteConfig,
) -> tuple[object, ...]:
    """订单缓存依赖 detector 参数和策略层门控参数；组合资金参数变化可复用。"""
    higher_timeframe = str(config.higher_timeframe).strip()
    return (
        _active_strategy_suite_cache_key(suite_config, include_trend_context=bool(higher_timeframe)),
        higher_timeframe,
        None if not higher_timeframe else config.higher_timeframe_max_age_minutes,
    )


def _active_strategy_suite_cache_key(
    suite_config: StrategySuiteConfig,
    *,
    include_trend_context: bool,
) -> tuple[object, ...]:
    """只把启用 detector 相关参数放入订单缓存键，避免未启用模块拖慢 sweep。"""
    enabled = tuple(suite_config.enabled)
    parts: list[object] = [
        ("enabled", enabled),
        ("risk_reward", float(suite_config.risk_reward)),
        ("max_holding_bars", int(suite_config.max_holding_bars)),
        ("max_actual_risk_pct", suite_config.max_actual_risk_pct),
        ("max_chase_pct", suite_config.max_chase_pct),
        ("side_mode", str(suite_config.side_mode)),
    ]
    for detector_name in enabled:
        parts.append((detector_name, _detector_cache_parameters(detector_name, suite_config)))
    if include_trend_context and "trend" not in enabled:
        parts.append(("higher_trend_context", _detector_cache_parameters("trend", suite_config)))
    return tuple(parts)


def _detector_cache_parameters(detector_name: str, suite_config: StrategySuiteConfig) -> tuple[object, ...]:
    if detector_name == "trend":
        return (
            int(suite_config.trend_lookback),
            float(suite_config.trend_min_score),
            float(suite_config.trend_strong_close_pos),
            float(suite_config.trend_min_body_ratio),
            int(suite_config.trend_pullback_lookback),
            int(suite_config.trend_h2_min_pullback_legs),
        )
    if detector_name == "range":
        return (
            int(suite_config.range_lookback),
            float(suite_config.range_middle_low),
            float(suite_config.range_middle_high),
            float(suite_config.range_false_break_buffer),
            float(suite_config.range_strong_close_pos),
            float(suite_config.range_min_score),
        )
    if detector_name == "channel":
        return (
            str(suite_config.channel_method),
            int(suite_config.channel_lookback),
            float(suite_config.channel_sigma_multiple),
            float(suite_config.channel_break_buffer),
            int(suite_config.channel_swing_left_bars),
            int(suite_config.channel_swing_right_bars),
        )
    if detector_name == "reversal":
        return (
            int(suite_config.reversal_lookback),
            float(suite_config.reversal_strong_close_pos),
            float(suite_config.reversal_min_body_ratio),
            float(suite_config.reversal_old_extreme_tolerance_pct),
            bool(suite_config.reversal_require_old_extreme_test),
            bool(suite_config.reversal_require_structure_confirmation),
        )
    raise ValueError(f"不支持的 detector：{detector_name}")


def _strategy_suite_config(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> StrategySuiteConfig:
    """把实验配置映射为订单生成配置；该对象可作为参数遍历的订单缓存键。"""
    enabled = config.detectors if isinstance(config, PortfolioExperimentConfig) else (config.detector,)
    return StrategySuiteConfig(
        enabled=enabled,
        risk_reward=config.risk_reward,
        max_holding_bars=config.max_holding_bars,
        max_actual_risk_pct=config.max_actual_risk_pct,
        max_chase_pct=config.max_chase_pct,
        side_mode=config.side_mode,
        trend_lookback=config.trend_lookback,
        trend_min_score=config.trend_min_score,
        trend_strong_close_pos=config.trend_strong_close_pos,
        trend_min_body_ratio=config.trend_min_body_ratio,
        trend_pullback_lookback=config.trend_pullback_lookback,
        trend_h2_min_pullback_legs=config.trend_h2_min_pullback_legs,
        range_lookback=config.range_lookback,
        range_middle_low=config.range_middle_low,
        range_middle_high=config.range_middle_high,
        range_false_break_buffer=config.range_false_break_buffer,
        range_strong_close_pos=config.range_strong_close_pos,
        range_min_score=config.range_min_score,
        channel_method=config.channel_method,
        channel_lookback=config.channel_lookback,
        channel_sigma_multiple=config.channel_sigma_multiple,
        channel_break_buffer=config.channel_break_buffer,
        channel_swing_left_bars=config.channel_swing_left_bars,
        channel_swing_right_bars=config.channel_swing_right_bars,
        reversal_lookback=config.reversal_lookback,
        reversal_strong_close_pos=config.reversal_strong_close_pos,
        reversal_min_body_ratio=config.reversal_min_body_ratio,
        reversal_old_extreme_tolerance_pct=config.reversal_old_extreme_tolerance_pct,
        reversal_require_old_extreme_test=config.reversal_require_old_extreme_test,
        reversal_require_structure_confirmation=config.reversal_require_structure_confirmation,
    )


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


def _symbol_name_map_for_config(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> dict[str, str]:
    metadata = _symbol_metadata_for_config(config)
    return {str(row.stock_code): str(row.stock_name) for row in metadata.itertuples(index=False)}


def _symbol_grouped_trade_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *,
    symbol_name_by_code: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """标的维度统计优先带股票名称；代码保留为复核键，不再让用户只看 stock_code。"""
    stats = _grouped_trade_statistics(trades, by="stock_code")
    if "stock_name" in stats.columns:
        return stats
    if "stock_code" not in stats.columns:
        return stats
    symbols = unique_symbols(tuple(config.symbols))
    name_by_symbol = symbol_name_by_code or _symbol_name_map_for_config(config)
    if stats.empty:
        return _zero_symbol_statistics(symbols, name_by_symbol)
    result = stats.copy()
    names = result["stock_code"].astype(str).map(lambda symbol: name_by_symbol.get(symbol, symbol))
    result.insert(0, "stock_name", names)
    existing_symbols = set(result["stock_code"].astype(str))
    missing = [symbol for symbol in symbols if symbol not in existing_symbols]
    if missing:
        result = pd.concat([result, _zero_symbol_statistics(missing, name_by_symbol)], ignore_index=True)
    symbol_order = {symbol: index for index, symbol in enumerate(symbols)}
    result["_symbol_order"] = result["stock_code"].astype(str).map(lambda symbol: symbol_order.get(symbol, len(symbol_order)))
    return (
        result.sort_values(["_symbol_order", "stock_code"], kind="mergesort")
        .drop(columns=["_symbol_order"])
        .reset_index(drop=True)
        .reindex(columns=pd.Index(["stock_name", "stock_code", *STAT_KEYS]))
    )


def _configured_detector_names(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> tuple[str, ...]:
    return tuple(config.detectors) if isinstance(config, PortfolioExperimentConfig) else (config.detector,)


def _zero_symbol_statistics(symbols: Sequence[str], symbol_name_by_code: Mapping[str, str]) -> pd.DataFrame:
    rows = [
        {
            "stock_name": symbol_name_by_code.get(symbol, symbol),
            "stock_code": symbol,
            **{key: 0.0 for key in STAT_KEYS},
        }
        for symbol in symbols
    ]
    return pd.DataFrame(rows, columns=pd.Index(["stock_name", "stock_code", *STAT_KEYS]))


def _case_setup_statistics(
    trades: pd.DataFrame,
    *,
    case_name: str,
    case_config_hash: str,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 case 汇总 setup 表现；没有成交但出现过信号的 setup 也保留零行。"""
    stats = _setup_trade_statistics(trades, order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=pd.Index(["case_name", "case_config_hash", *SETUP_STAT_FIELDS, *STAT_KEYS]))
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats


def _case_strategy_statistics(
    trades: pd.DataFrame,
    strategies: Sequence[object],
    *,
    case_name: str,
    case_config_hash: str,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 case 汇总策略表现；已启用但没有成交的策略也保留零行。"""
    columns = pd.Index(["case_name", "case_config_hash", "strategy_name", *STAT_KEYS])
    stats = _strategy_trade_statistics(trades, strategies, order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=columns)
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats.reindex(columns=columns)


def _case_detector_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *,
    case_name: str,
    case_config_hash: str,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 case 汇总识别模块表现；已启用但没有成交的 detector 也保留零行。"""
    columns = pd.Index(["case_name", "case_config_hash", "detector_name", *STAT_KEYS])
    stats = _detector_trade_statistics(trades, _configured_detector_names(config), order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=columns)
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats.reindex(columns=columns)


def _case_symbol_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *,
    symbol_name_by_code: Mapping[str, str],
    case_name: str,
    case_config_hash: str,
) -> pd.DataFrame:
    """按 case 汇总标的表现；没有成交的样本股票也保留零值行。"""
    stats = _symbol_grouped_trade_statistics(
        trades,
        config,
        symbol_name_by_code=symbol_name_by_code,
    )
    if stats.empty:
        return pd.DataFrame(columns=pd.Index(["case_name", "case_config_hash", "stock_name", "stock_code", *STAT_KEYS]))
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats


def _concat_case_strategy_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "strategy_name", *STAT_KEYS])
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _concat_case_detector_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "detector_name", *STAT_KEYS])
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _case_decision_statistics(
    decisions: pd.DataFrame,
    *,
    case_name: str,
    case_config_hash: str,
    group_fields: tuple[str, ...],
) -> pd.DataFrame:
    """按 case 汇总 setup 决策分布；用于解释参数组的拒绝结构。"""
    stats = compute_decision_reason_statistics(decisions, group_fields=group_fields)
    columns = _case_decision_columns(group_fields)
    if stats.empty:
        return pd.DataFrame(columns=columns)
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats.reindex(columns=columns)


def _concat_case_setup_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", *SETUP_STAT_FIELDS, *STAT_KEYS])
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _concat_case_symbol_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "stock_name", "stock_code", *STAT_KEYS])
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _concat_case_decision_statistics(frames: Sequence[pd.DataFrame], *, group_fields: tuple[str, ...]) -> pd.DataFrame:
    columns = _case_decision_columns(group_fields)
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _ranked_case_strategy_statistics(case_strategy: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if case_strategy.empty:
        return pd.DataFrame(columns=pd.Index(SWEEP_CASE_STRATEGY_COLUMNS))
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_strategy.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return (
        merged.reindex(columns=pd.Index(SWEEP_CASE_STRATEGY_COLUMNS))
        .sort_values(["sweep_rank", "case_name", "strategy_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def _ranked_case_detector_statistics(case_detector: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if case_detector.empty:
        return pd.DataFrame(columns=pd.Index(SWEEP_CASE_DETECTOR_COLUMNS))
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_detector.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return (
        merged.reindex(columns=pd.Index(SWEEP_CASE_DETECTOR_COLUMNS))
        .sort_values(["sweep_rank", "case_name", "detector_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def _ranked_case_setup_statistics(case_setup: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if case_setup.empty:
        return pd.DataFrame(columns=pd.Index(SWEEP_CASE_SETUP_COLUMNS))
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_setup.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return (
        merged.reindex(columns=pd.Index(SWEEP_CASE_SETUP_COLUMNS))
        .sort_values(["sweep_rank", "case_name", *SETUP_STAT_FIELDS], kind="mergesort")
        .reset_index(drop=True)
    )


def _ranked_case_symbol_statistics(case_symbol: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    if case_symbol.empty:
        return pd.DataFrame(columns=pd.Index(SWEEP_CASE_SYMBOL_COLUMNS))
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_symbol.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return (
        merged.reindex(columns=pd.Index(SWEEP_CASE_SYMBOL_COLUMNS))
        .sort_values(["sweep_rank", "case_name", "stock_name", "stock_code"], kind="mergesort")
        .reset_index(drop=True)
    )


def _ranked_case_decision_statistics(
    case_decisions: pd.DataFrame,
    table: pd.DataFrame,
    *,
    group_fields: tuple[str, ...],
) -> pd.DataFrame:
    columns = _ranked_case_decision_columns(group_fields)
    if case_decisions.empty:
        return pd.DataFrame(columns=columns)
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_decisions.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    sort_columns = ["sweep_rank", "case_name", *group_fields, "status", "reason"]
    return (
        merged.reindex(columns=columns)
        .sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )


def _case_decision_columns(group_fields: tuple[str, ...]) -> pd.Index:
    stats_columns = compute_decision_reason_statistics(pd.DataFrame(), group_fields=group_fields).columns
    return pd.Index(["case_name", "case_config_hash", *stats_columns])


def _ranked_case_decision_columns(group_fields: tuple[str, ...]) -> pd.Index:
    return pd.Index(["sweep_rank", "pareto_rank", "is_pareto_efficient", *_case_decision_columns(group_fields)])


def _rank_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    return rank_sweep_table(table)


def _pareto_front_ranks(table: pd.DataFrame) -> list[int]:
    return _build_pareto_front_ranks(table, dominance_matrix_fn=_pareto_dominance_matrix)


def _pareto_dominance_matrix(values: np.ndarray) -> np.ndarray:
    return _build_pareto_dominance_matrix(values)


def _pareto_score_table(table: pd.DataFrame) -> pd.DataFrame:
    return _build_pareto_score_table(table)
