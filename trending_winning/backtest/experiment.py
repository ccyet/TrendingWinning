from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from trending_winning.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    run_order_backtest_from_normalized,
    run_single_strategy_backtest_from_normalized,
)
from trending_winning.backtest.periods import compute_period_return_statistics, compute_period_returns
from trending_winning.backtest.portfolio import (
    PortfolioConfig,
    PortfolioCandidateSet,
    prepare_portfolio_candidates_from_normalized,
    run_portfolio_candidate_backtest_from_normalized,
    run_portfolio_backtest_from_normalized,
)
from trending_winning.backtest.stats import (
    STAT_KEYS,
    compute_decision_reason_statistics,
    compute_grouped_trade_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.data.repository import MarketDataRepository
from trending_winning.data.schema import unique_symbols
from trending_winning.data.summary import DATA_INVENTORY_SUMMARY_KEYS, summarize_data_management
from trending_winning.data.symbols import DEFAULT_STOCK_NAME_BY_CODE, SYMBOL_METADATA_COLUMNS, load_symbol_metadata
from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy, TimeframeAlignmentConfig
from trending_winning.strategies.runtime import execute_strategy, execute_strategies
from trending_winning.strategies.suite import StrategySuiteConfig, create_default_strategy_suite, create_strategy_for_detector

DATA_SCOPE_SWEEP_FIELDS = {
    "data_root",
    "symbols",
    "timeframe",
    "higher_timeframe",
    "start",
    "end",
    "adjust",
    "strict_data_quality",
    "min_coverage_ratio",
}

SWEEP_PARETO_OBJECTIVES = (
    ("total_return", "max"),
    ("max_drawdown", "max"),
    ("ulcer_index", "min"),
    ("monthly_worst_return", "max"),
    ("monthly_return_std", "min"),
    ("trade_count", "max"),
)

DETECTOR_PARAMETER_FIELDS = {
    "trend": frozenset(
        {
            "trend_lookback",
            "trend_min_score",
            "trend_strong_close_pos",
            "trend_min_body_ratio",
            "trend_pullback_lookback",
            "trend_h2_min_pullback_legs",
        }
    ),
    "range": frozenset(
        {
            "range_lookback",
            "range_middle_low",
            "range_middle_high",
            "range_false_break_buffer",
            "range_strong_close_pos",
            "range_min_score",
        }
    ),
    "channel": frozenset(
        {
            "channel_method",
            "channel_lookback",
            "channel_sigma_multiple",
            "channel_break_buffer",
            "channel_swing_left_bars",
            "channel_swing_right_bars",
        }
    ),
    "reversal": frozenset(
        {
            "reversal_lookback",
            "reversal_strong_close_pos",
            "reversal_min_body_ratio",
            "reversal_old_extreme_tolerance_pct",
            "reversal_require_old_extreme_test",
            "reversal_require_structure_confirmation",
        }
    ),
}

ALL_DETECTOR_PARAMETER_FIELDS = frozenset().union(*DETECTOR_PARAMETER_FIELDS.values())
NON_REPRODUCIBLE_CONFIG_HASH_FIELDS = frozenset({"name", "data_root", "output_dir"})

SWEEP_SUMMARY_CONTEXT_COLUMNS = (
    *DATA_INVENTORY_SUMMARY_KEYS,
    "data_audit_row_count",
    "data_audit_ok_count",
    "data_audit_failed_count",
    "data_missing_rows",
    "data_expected_rows",
    "data_weighted_coverage_ratio",
    "data_coverage_p05",
    "data_coverage_p50",
    "data_coverage_p95",
    "data_max_missing_gap_minutes",
    "data_max_missing_gap_start_at",
    "data_max_missing_gap_end_at",
    "data_min_coverage_threshold",
    "data_coverage_below_min_count",
    "data_coverage_below_min_ratio",
    "limit_filter_symbol_count",
    "limit_filter_trading_days",
    "limit_filter_filtered_days",
    "limit_filter_rows_before",
    "limit_filter_rows_after",
    "filtered_limit_open_count",
)

SWEEP_SUMMARY_BEST_COLUMNS = (
    "sweep_rank",
    "pareto_rank",
    "is_pareto_efficient",
    "total_return",
    "annualized_return",
    "max_drawdown",
    "equity_sharpe",
    "calmar_ratio",
    "ulcer_index",
    "trade_count",
    "monthly_count",
    "monthly_win_rate",
    "monthly_worst_return",
    "monthly_return_std",
    "monthly_max_consecutive_losses",
    "monthly_max_recovery_periods",
)

PARAMETER_SUMMARY_METRICS = (
    ("total_return", "avg_total_return", "mean"),
    ("total_return", "median_total_return", "median"),
    ("max_drawdown", "avg_max_drawdown", "mean"),
    ("monthly_worst_return", "avg_monthly_worst_return", "mean"),
    ("monthly_return_std", "avg_monthly_return_std", "mean"),
    ("monthly_max_consecutive_losses", "avg_monthly_max_consecutive_losses", "mean"),
    ("monthly_max_recovery_periods", "avg_monthly_max_recovery_periods", "mean"),
    ("trade_count", "avg_trade_count", "mean"),
    ("return_per_exposure_bar", "avg_return_per_exposure_bar", "mean"),
    ("return_per_capital_exposure_bar", "avg_return_per_capital_exposure_bar", "mean"),
    ("return_per_margin_exposure_bar", "avg_return_per_margin_exposure_bar", "mean"),
    ("take_profit_exit_rate", "avg_take_profit_exit_rate", "mean"),
    ("trailing_take_profit_exit_rate", "avg_trailing_take_profit_exit_rate", "mean"),
    ("stop_loss_exit_rate", "avg_stop_loss_exit_rate", "mean"),
    ("max_holding_exit_rate", "avg_max_holding_exit_rate", "mean"),
    ("bars_per_second", "avg_bars_per_second", "mean"),
    ("acceptance_rate", "avg_acceptance_rate", "mean"),
    ("rejection_rate", "avg_rejection_rate", "mean"),
    ("rejected_no_fill_count", "avg_rejected_no_fill_count", "mean"),
    ("avg_accepted_actual_risk_pct", "avg_accepted_actual_risk_pct", "mean"),
    ("avg_accepted_actual_chase_pct", "avg_accepted_actual_chase_pct", "mean"),
    ("avg_accepted_actual_reward_to_risk", "avg_accepted_actual_reward_to_risk", "mean"),
    ("avg_executed_actual_risk_pct", "avg_executed_actual_risk_pct", "mean"),
    ("avg_executed_actual_chase_pct", "avg_executed_actual_chase_pct", "mean"),
    ("avg_executed_actual_reward_to_risk", "avg_executed_actual_reward_to_risk", "mean"),
    ("strategy_filter_acceptance_rate", "avg_strategy_filter_acceptance_rate", "mean"),
    ("strategy_filter_rejection_rate", "avg_strategy_filter_rejection_rate", "mean"),
)

SETUP_STAT_FIELDS = ("detector_name", "event_type", "side")
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


@dataclass(frozen=True)
class PortfolioExperimentConfig:
    """组合回测实验配置；用于复现实验和保存产物。"""

    name: str
    data_root: str
    symbols: tuple[str, ...]
    timeframe: str
    start: str
    end: str
    adjust: str = "qfq"
    higher_timeframe: str = ""
    higher_timeframe_max_age_minutes: int | None = None
    detectors: tuple[str, ...] = ("trend", "range", "channel")
    risk_reward: float = 2.0
    max_holding_bars: int = 12
    max_actual_risk_pct: float | None = None
    max_chase_pct: float | None = None
    side_mode: str = "both"
    max_open_positions: int = 5
    capital_per_trade: float | None = None
    risk_per_trade: float | None = None
    max_capital_per_trade: float = 1.0
    short_margin_rate: float = 1.0
    reserve_cash: float = 0.0
    allow_same_symbol_overlap: bool = False
    strategy_priority: dict[str, int] = field(default_factory=dict)
    strategy_capital_limit: dict[str, float] = field(default_factory=dict)
    sector_capital_limit: dict[str, float] = field(default_factory=dict)
    symbol_sector_map: dict[str, str] = field(default_factory=dict)
    sector_metadata_key: str = "sector"
    default_sector: str = "UNKNOWN"
    intrabar_exit_policy: str = "conservative"
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    initial_equity: float = 1.0
    trailing_take_profit_activation_pct: float = 0.0
    trailing_take_profit_drawdown_pct: float = 0.0
    trailing_take_profit_ma_period: int = 0
    strict_data_quality: bool = True
    min_coverage_ratio: float | None = None
    output_dir: str = ""
    trend_lookback: int = 20
    trend_min_score: float = 1.0
    trend_strong_close_pos: float = 0.65
    trend_min_body_ratio: float = 0.45
    trend_pullback_lookback: int = 5
    trend_h2_min_pullback_legs: int = 2
    range_lookback: int = 20
    range_middle_low: float = 0.25
    range_middle_high: float = 0.75
    range_false_break_buffer: float = 0.0
    range_strong_close_pos: float = 0.65
    range_min_score: float = 0.8
    channel_method: str = "regression"
    channel_lookback: int = 40
    channel_sigma_multiple: float = 2.0
    channel_break_buffer: float = 0.0
    channel_swing_left_bars: int = 2
    channel_swing_right_bars: int = 2
    reversal_lookback: int = 20
    reversal_strong_close_pos: float = 0.65
    reversal_min_body_ratio: float = 0.45
    reversal_old_extreme_tolerance_pct: float = 0.01
    reversal_require_old_extreme_test: bool = True
    reversal_require_structure_confirmation: bool = True


@dataclass(frozen=True)
class SingleStrategyExperimentConfig:
    """单策略实验配置；只绑定一个 detector，不进入组合仓位分配层。"""

    name: str
    data_root: str
    symbols: tuple[str, ...]
    timeframe: str
    start: str
    end: str
    detector: str
    adjust: str = "qfq"
    higher_timeframe: str = ""
    higher_timeframe_max_age_minutes: int | None = None
    risk_reward: float = 2.0
    max_holding_bars: int = 12
    max_actual_risk_pct: float | None = None
    max_chase_pct: float | None = None
    side_mode: str = "both"
    intrabar_exit_policy: str = "conservative"
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    initial_equity: float = 1.0
    trailing_take_profit_activation_pct: float = 0.0
    trailing_take_profit_drawdown_pct: float = 0.0
    trailing_take_profit_ma_period: int = 0
    strict_data_quality: bool = True
    min_coverage_ratio: float | None = None
    output_dir: str = ""
    trend_lookback: int = 20
    trend_min_score: float = 1.0
    trend_strong_close_pos: float = 0.65
    trend_min_body_ratio: float = 0.45
    trend_pullback_lookback: int = 5
    trend_h2_min_pullback_legs: int = 2
    range_lookback: int = 20
    range_middle_low: float = 0.25
    range_middle_high: float = 0.75
    range_false_break_buffer: float = 0.0
    range_strong_close_pos: float = 0.65
    range_min_score: float = 0.8
    channel_method: str = "regression"
    channel_lookback: int = 40
    channel_sigma_multiple: float = 2.0
    channel_break_buffer: float = 0.0
    channel_swing_left_bars: int = 2
    channel_swing_right_bars: int = 2
    reversal_lookback: int = 20
    reversal_strong_close_pos: float = 0.65
    reversal_min_body_ratio: float = 0.45
    reversal_old_extreme_tolerance_pct: float = 0.01
    reversal_require_old_extreme_test: bool = True
    reversal_require_structure_confirmation: bool = True


@dataclass(frozen=True)
class PortfolioExperimentResult:
    """组合实验结果；保留配置、回测结果和数据过滤信息。"""

    config: PortfolioExperimentConfig
    backtest: BacktestResult
    input_bar_count: int
    filtered_limit_open_count: int
    data_coverage: pd.DataFrame
    strategy_stats: pd.DataFrame
    symbol_stats: pd.DataFrame
    side_stats: pd.DataFrame
    exit_reason_stats: pd.DataFrame
    monthly_returns: pd.DataFrame
    elapsed_seconds: float
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_inventory: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class SingleStrategyExperimentResult:
    """单策略实验结果；不生成组合层持仓分配产物。"""

    config: SingleStrategyExperimentConfig
    backtest: BacktestResult
    input_bar_count: int
    filtered_limit_open_count: int
    elapsed_seconds: float
    data_coverage: pd.DataFrame
    strategy_stats: pd.DataFrame
    symbol_stats: pd.DataFrame
    side_stats: pd.DataFrame
    exit_reason_stats: pd.DataFrame
    monthly_returns: pd.DataFrame
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_inventory: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class PortfolioBenchmarkReport:
    """组合回测性能报告；记录吞吐和产出规模。"""

    experiment_name: str
    bar_count: int
    trade_count: int
    equity_points: int
    elapsed_seconds: float
    bars_per_second: float
    trades_per_second: float


@dataclass(frozen=True)
class PortfolioSweepResult:
    """参数遍历结果；一次加载数据后复用 K 线跑多组组合参数。"""

    config: PortfolioExperimentConfig
    grid: dict[str, list[object]]
    table: pd.DataFrame
    data_coverage: pd.DataFrame
    input_bar_count: int
    filtered_limit_open_count: int
    elapsed_seconds: float
    strategy_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    symbol_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_inventory: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class SingleStrategySweepResult:
    """单策略参数遍历结果；不进入组合仓位分配层。"""

    config: SingleStrategyExperimentConfig
    grid: dict[str, list[object]]
    table: pd.DataFrame
    data_coverage: pd.DataFrame
    input_bar_count: int
    filtered_limit_open_count: int
    elapsed_seconds: float
    strategy_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    symbol_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_inventory: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _LoadedExperimentData:
    """实验装载后的标准数据包；集中携带主周期、高周期和审计结果。"""

    bars: pd.DataFrame
    higher_bars: pd.DataFrame
    data_audit: pd.DataFrame
    data_inventory: pd.DataFrame
    limit_filter_audit: pd.DataFrame
    filtered_limit_open_days: pd.DataFrame


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
    monthly_returns = compute_period_returns(_trade_dated_equity_curve(backtest), freq="M")
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
            config,
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
            config,
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


def _load_experiment_data(
    repo: MarketDataRepository,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> _LoadedExperimentData:
    higher_timeframe = str(config.higher_timeframe).strip()
    data_inventory = repo.inventory(
        timeframes=tuple(_experiment_inventory_timeframes(config)),
        symbols=config.symbols,
    )
    if higher_timeframe:
        if higher_timeframe == config.timeframe:
            raise ValueError("higher_timeframe 不能和 timeframe 相同。")
        bundle = repo.load_multi_timeframe_backtest_data(
            timeframes=(config.timeframe, higher_timeframe),
            symbols=config.symbols,
            start=config.start,
            end=config.end,
            strict_data_quality=config.strict_data_quality,
            min_coverage_ratio=config.min_coverage_ratio,
        )
        return _LoadedExperimentData(
            bars=bundle.bars_by_timeframe.get(config.timeframe, pd.DataFrame()),
            higher_bars=bundle.bars_by_timeframe.get(higher_timeframe, pd.DataFrame()),
            data_audit=bundle.data_audit,
            data_inventory=data_inventory,
            limit_filter_audit=bundle.limit_filter_audit,
            filtered_limit_open_days=bundle.filtered_limit_open_days,
        )
    bundle = repo.load_backtest_data(
        timeframe=config.timeframe,
        symbols=config.symbols,
        start=config.start,
        end=config.end,
        strict_data_quality=config.strict_data_quality,
        min_coverage_ratio=config.min_coverage_ratio,
    )
    return _LoadedExperimentData(
        bars=bundle.bars,
        higher_bars=pd.DataFrame(),
        data_audit=bundle.data_audit,
        data_inventory=data_inventory,
        limit_filter_audit=bundle.limit_filter_audit,
        filtered_limit_open_days=bundle.filtered_limit_open_days,
    )


def _experiment_inventory_timeframes(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> list[str]:
    """实验结果需要保存日 K 依赖、主周期和可选高周期的本地缓存快照。"""
    timeframes = ["1d", str(config.timeframe)]
    higher_timeframe = str(config.higher_timeframe).strip()
    if higher_timeframe and higher_timeframe not in timeframes:
        timeframes.append(higher_timeframe)
    return timeframes


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


def _with_data_management_statistics(
    result: BacktestResult,
    data: _LoadedExperimentData,
    *,
    min_coverage_ratio: float | None,
) -> BacktestResult:
    """运行态结果也携带数据审计摘要，保证 Web/CLI 和保存产物统计口径一致。"""
    stats = dict(result.stats)
    stats.update(
        summarize_data_management(
            data.data_audit,
            data.limit_filter_audit,
            filtered_limit_open_count=len(data.filtered_limit_open_days),
            data_inventory=data.data_inventory,
            min_coverage_ratio=min_coverage_ratio,
        )
    )
    return BacktestResult(
        trades=result.trades,
        equity_curve=result.equity_curve,
        stats=stats,
        order_decisions=result.order_decisions,
        strategy_filter_decisions=result.strategy_filter_decisions,
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
    equity_curve = _trade_dated_equity_curve(result) if use_trade_dates else result.equity_curve
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


def _symbol_metadata_for_config(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> pd.DataFrame:
    """把实验涉及的股票名称随结果一起保存，避免统计表脱离代码名称映射。"""
    metadata = load_symbol_metadata(config.data_root)
    metadata_by_symbol = {str(row.stock_code): row for row in metadata.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(tuple(config.symbols)):
        if symbol in metadata_by_symbol:
            record = metadata_by_symbol[symbol]
            rows.append(
                {
                    "stock_code": symbol,
                    "stock_name": str(record.stock_name),
                    "source": str(record.source),
                    "path": str(record.path),
                }
            )
            continue
        name = DEFAULT_STOCK_NAME_BY_CODE.get(symbol)
        if name:
            rows.append({"stock_code": symbol, "stock_name": name, "source": "default_builtin", "path": ""})
    return pd.DataFrame(rows, columns=pd.Index(SYMBOL_METADATA_COLUMNS))


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


def save_single_strategy_experiment(result: SingleStrategyExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(_json_dump(_json_ready(asdict(result.config))))
    stats = dict(result.backtest.stats)
    stats.update(
        summarize_data_management(
            result.data_coverage,
            result.limit_filter_audit,
            filtered_limit_open_count=result.filtered_limit_open_count,
            data_inventory=result.data_inventory,
            min_coverage_ratio=result.config.min_coverage_ratio,
        )
    )
    stats.update(compute_period_return_statistics(result.monthly_returns, prefix="monthly"))
    stats["elapsed_seconds"] = float(result.elapsed_seconds)
    (output_dir / "stats.json").write_text(_json_dump(_json_ready(stats)))
    result.backtest.trades.to_csv(output_dir / "trades.csv", index=False)
    result.backtest.order_decisions.to_csv(output_dir / "order_decisions.csv", index=False)
    result.backtest.strategy_filter_decisions.to_csv(output_dir / "strategy_filter_decisions.csv", index=False)
    result.backtest.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    _symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "setup_strategy_filter_stats.csv", index=False)
    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    return output_dir


def save_portfolio_experiment(result: PortfolioExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(_json_dump(_json_ready(asdict(result.config))))
    stats = dict(result.backtest.stats)
    stats.update(
        summarize_data_management(
            result.data_coverage,
            result.limit_filter_audit,
            filtered_limit_open_count=result.filtered_limit_open_count,
            data_inventory=result.data_inventory,
            min_coverage_ratio=result.config.min_coverage_ratio,
        )
    )
    stats.update(compute_period_return_statistics(result.monthly_returns, prefix="monthly"))
    stats["elapsed_seconds"] = float(result.elapsed_seconds)
    (output_dir / "stats.json").write_text(_json_dump(_json_ready(stats)))
    result.backtest.trades.to_csv(output_dir / "trades.csv", index=False)
    result.backtest.order_decisions.to_csv(output_dir / "order_decisions.csv", index=False)
    result.backtest.strategy_filter_decisions.to_csv(output_dir / "strategy_filter_decisions.csv", index=False)
    result.backtest.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    _symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "setup_strategy_filter_stats.csv", index=False)
    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    return output_dir


def save_portfolio_benchmark(config: PortfolioExperimentConfig, report: PortfolioBenchmarkReport) -> Path:
    output_dir = Path(config.output_dir or f"runs/{config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(_json_dump(_json_ready(asdict(report))))
    return output_dir


def save_portfolio_sweep(result: PortfolioSweepResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = _json_ready(asdict(result.config))
    config_payload["sweep_grid"] = _json_ready(result.grid)
    (output_dir / "config.json").write_text(_json_dump(config_payload))
    (output_dir / "summary.json").write_text(_json_dump(_json_ready(_sweep_summary_statistics(result))))
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    _pareto_sweep_table(result.table).to_csv(output_dir / "pareto.csv", index=False)
    _parameter_summary_table(result).to_csv(output_dir / "parameter_summary.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "case_strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "case_detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "case_setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "case_symbol_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "case_setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "case_setup_strategy_filter_stats.csv", index=False)
    _write_jsonl(output_dir / "case_configs.jsonl", _sweep_case_config_records(result))
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    _symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    return output_dir


def save_single_strategy_sweep(result: SingleStrategySweepResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = _json_ready(asdict(result.config))
    config_payload["sweep_grid"] = _json_ready(result.grid)
    (output_dir / "config.json").write_text(_json_dump(config_payload))
    (output_dir / "summary.json").write_text(_json_dump(_json_ready(_sweep_summary_statistics(result))))
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    _pareto_sweep_table(result.table).to_csv(output_dir / "pareto.csv", index=False)
    _parameter_summary_table(result).to_csv(output_dir / "parameter_summary.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "case_strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "case_detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "case_setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "case_symbol_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "case_setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "case_setup_strategy_filter_stats.csv", index=False)
    _write_jsonl(output_dir / "case_configs.jsonl", _sweep_case_config_records(result))
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    _symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    return output_dir


def _sweep_summary_statistics(result: PortfolioSweepResult | SingleStrategySweepResult) -> dict[str, object]:
    """把参数遍历压成一份总览 JSON，便于 Web/CLI 快速展示，不再先扫完整 CSV。"""
    table = result.table
    summary: dict[str, object] = {
        "case_count": int(len(table)),
        "grid_case_count": int(math.prod(_grid_value_counts(result.grid).values())) if result.grid else 0,
        "grid_field_count": int(len(result.grid)),
        "grid_fields": list(result.grid),
        "grid_value_counts": _grid_value_counts(result.grid),
        "pareto_case_count": _truthy_column_count(table, "is_pareto_efficient"),
        "elapsed_seconds": float(result.elapsed_seconds),
        "input_bar_count": int(result.input_bar_count),
        "filtered_limit_open_count": int(result.filtered_limit_open_count),
        "best_case_name": "",
        "best_case_config_hash": "",
    }
    if not table.empty:
        best = table.iloc[0]
        summary["best_case_name"] = str(best.get("case_name", ""))
        summary["best_case_config_hash"] = str(best.get("case_config_hash", ""))
        for column in SWEEP_SUMMARY_BEST_COLUMNS:
            if column in table.columns:
                summary[f"best_{column}"] = _json_scalar(best[column])
        for column in SWEEP_SUMMARY_CONTEXT_COLUMNS:
            if column in table.columns:
                summary[column] = _json_scalar(best[column])

    summary.update(_cache_status_statistics(table, "order_cache_status", prefix="order_cache"))
    summary.update(_cache_status_statistics(table, "candidate_cache_status", prefix="candidate_cache"))
    summary.update(_case_trade_summary_statistics(result.strategy_stats, prefix="case_strategy"))
    summary.update(_case_trade_summary_statistics(result.detector_stats, prefix="case_detector"))
    summary.update(_case_trade_summary_statistics(result.setup_stats, prefix="case_setup"))
    summary.update(_case_trade_summary_statistics(result.symbol_stats, prefix="case_symbol"))
    summary.update(
        _case_decision_summary_statistics(
            result.setup_order_decision_stats,
            prefix="case_setup_order",
        )
    )
    summary.update(
        _case_decision_summary_statistics(
            result.setup_strategy_filter_stats,
            prefix="case_setup_strategy_filter",
        )
    )
    for column in ("generated_order_count", "candidate_count", "candidate_rejection_count"):
        if column in table.columns:
            summary[column] = _numeric_column_sum(table, column)
    return summary


def _pareto_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    """提取第一层 Pareto 候选，保持 sweep.csv 的列和排名顺序。"""
    if table.empty or "pareto_rank" not in table.columns:
        return table.iloc[0:0].copy()
    ranks = pd.to_numeric(table["pareto_rank"], errors="coerce")
    return table.loc[ranks.eq(1)].copy()


def _parameter_summary_table(result: PortfolioSweepResult | SingleStrategySweepResult) -> pd.DataFrame:
    """按参数字段和值聚合 sweep 表，帮助判断哪些参数区间更稳。"""
    columns = [
        "parameter",
        "value",
        "case_count",
        "pareto_case_count",
        "pareto_hit_rate",
        "positive_return_case_count",
        "positive_return_rate",
        "std_total_return",
        "best_total_return",
        "worst_total_return",
        "best_sweep_rank",
        "best_case_name",
        "best_case_config_hash",
        *[output for _, output, _ in PARAMETER_SUMMARY_METRICS],
    ]
    table = result.table
    if table.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for parameter in result.grid:
        if parameter not in table.columns:
            continue
        labels = table[parameter].map(_parameter_value_label)
        for value, group in table.groupby(labels, sort=False, dropna=False):
            best = group.sort_values("sweep_rank", ascending=True, kind="mergesort").iloc[0]
            case_count = int(len(group))
            pareto_case_count = _truthy_column_count(group, "is_pareto_efficient")
            row: dict[str, object] = {
                "parameter": str(parameter),
                "value": str(value),
                "case_count": case_count,
                "pareto_case_count": pareto_case_count,
                **_parameter_robustness_metrics(group, case_count=case_count, pareto_case_count=pareto_case_count),
                "best_sweep_rank": _json_scalar(best.get("sweep_rank")),
                "best_case_name": str(best.get("case_name", "")),
                "best_case_config_hash": str(best.get("case_config_hash", "")),
            }
            for source, output, method in PARAMETER_SUMMARY_METRICS:
                row[output] = _numeric_group_metric(group, source, method)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    summary = pd.DataFrame(rows, columns=columns)
    return summary.sort_values(["parameter", "best_sweep_rank", "value"], kind="mergesort").reset_index(drop=True)


def _parameter_robustness_metrics(
    group: pd.DataFrame,
    *,
    case_count: int,
    pareto_case_count: int,
) -> dict[str, float]:
    """汇总单个参数值的稳健性，避免只按平均收益选参数。"""
    total_return = _numeric_group_values(group, "total_return")
    positive_return_count = int(total_return.gt(0).sum()) if not total_return.empty else 0
    return {
        "pareto_hit_rate": float(pareto_case_count / case_count) if case_count else 0.0,
        "positive_return_case_count": float(positive_return_count),
        "positive_return_rate": float(positive_return_count / case_count) if case_count else 0.0,
        "std_total_return": float(total_return.std(ddof=0)) if not total_return.empty else 0.0,
        "best_total_return": float(total_return.max()) if not total_return.empty else 0.0,
        "worst_total_return": float(total_return.min()) if not total_return.empty else 0.0,
    }


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
    stats = _detector_trade_statistics(trades, config, order_decisions, filter_decisions)
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


def _numeric_group_metric(group: pd.DataFrame, column: str, method: str) -> float:
    values = _numeric_group_values(group, column)
    if values.empty:
        return 0.0
    if method == "median":
        return float(values.median())
    return float(values.mean())


def _numeric_group_values(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(group[column], errors="coerce").dropna()


def _parameter_value_label(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, dict | list | tuple):
        return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if pd.isna(value):
        return ""
    return str(value)


def _grid_value_counts(grid: Mapping[str, Sequence[object]]) -> dict[str, int]:
    return {str(key): len(list(values)) for key, values in grid.items()}


def _cache_status_statistics(table: pd.DataFrame, column: str, *, prefix: str) -> dict[str, float]:
    keys = {
        f"{prefix}_hit_count": 0.0,
        f"{prefix}_miss_count": 0.0,
        f"{prefix}_hit_rate": 0.0,
    }
    if table.empty or column not in table.columns:
        return keys
    status = table[column].fillna("").astype(str)
    hits = float(status.eq("hit").sum())
    misses = float(status.eq("miss").sum())
    total = hits + misses
    keys[f"{prefix}_hit_count"] = hits
    keys[f"{prefix}_miss_count"] = misses
    keys[f"{prefix}_hit_rate"] = float(hits / total) if total else 0.0
    return keys


def _case_trade_summary_statistics(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    """把 case 级绩效表压成 summary 字段，避免总览页先读取大 CSV。"""
    keys = {
        f"{prefix}_row_count": 0.0,
        f"{prefix}_trade_count": 0.0,
        f"{prefix}_zero_trade_row_count": 0.0,
    }
    if frame.empty or "trade_count" not in frame.columns:
        return keys
    trades = pd.to_numeric(frame["trade_count"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_row_count": float(len(frame)),
        f"{prefix}_trade_count": float(trades.sum()),
        f"{prefix}_zero_trade_row_count": float(trades.eq(0.0).sum()),
    }


def _case_decision_summary_statistics(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
    """把 case 级决策明细压成 summary 字段，供 Web 总览不扫大 CSV。"""
    keys = {
        f"{prefix}_decision_row_count": 0.0,
        f"{prefix}_decision_count": 0.0,
        f"{prefix}_accepted_count": 0.0,
        f"{prefix}_rejected_count": 0.0,
        f"{prefix}_rejection_rate": 0.0,
    }
    if frame.empty or "decision_count" not in frame.columns:
        return keys

    decisions = pd.to_numeric(frame["decision_count"], errors="coerce").fillna(0.0)
    status = (
        frame["status"].fillna("").astype(str)
        if "status" in frame.columns
        else pd.Series([""] * len(frame), index=frame.index, dtype=str)
    )
    total = float(decisions.sum())
    accepted = float(decisions.loc[status.eq("accepted")].sum())
    rejected = float(decisions.loc[status.eq("rejected")].sum())
    return {
        f"{prefix}_decision_row_count": float(len(frame)),
        f"{prefix}_decision_count": total,
        f"{prefix}_accepted_count": accepted,
        f"{prefix}_rejected_count": rejected,
        f"{prefix}_rejection_rate": float(rejected / total) if total else 0.0,
    }


def _truthy_column_count(table: pd.DataFrame, column: str) -> int:
    if table.empty or column not in table.columns:
        return 0
    values = table[column]
    if pd.api.types.is_bool_dtype(values):
        return int(values.fillna(False).sum())
    normalized = values.fillna(False).astype(str).str.lower()
    return int(normalized.isin(("true", "1", "yes")).sum())


def _numeric_column_sum(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return 0.0
    return float(pd.to_numeric(table[column], errors="coerce").fillna(0.0).sum())


def _json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, int | float | str | bool):
        return value
    if pd.isna(value):
        return None
    return value


def _sweep_variants(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    grid: Mapping[str, Sequence[object]],
) -> list[PortfolioExperimentConfig | SingleStrategyExperimentConfig]:
    if not grid:
        raise ValueError("grid 不能为空。")
    config_fields = {field.name for field in fields(type(config))}
    unknown = set(grid).difference(config_fields)
    if unknown:
        raise ValueError(f"grid 包含不支持的配置字段：{', '.join(sorted(unknown))}")
    data_scope_fields = set(grid).intersection(DATA_SCOPE_SWEEP_FIELDS)
    if data_scope_fields:
        raise ValueError(f"不能在同一次 sweep 中改变数据范围字段：{', '.join(sorted(data_scope_fields))}")
    raw_keys = list(grid)
    raw_value_lists = [list(grid[key]) for key in raw_keys]
    empty_keys = [key for key, values in zip(raw_keys, raw_value_lists, strict=False) if not values]
    if empty_keys:
        raise ValueError(f"grid 字段不能为空：{', '.join(empty_keys)}")
    effective_grid = _effective_sweep_grid(config, grid)
    if not effective_grid:
        return [config]
    keys = list(effective_grid)
    raw_value_lists = [list(effective_grid[key]) for key in keys]
    value_lists = [_deduplicate_sweep_grid_values(values) for values in raw_value_lists]
    variants = [
        replace(config, **dict(zip(keys, values, strict=False)))
        for values in product(*value_lists)
    ]
    return _deduplicate_sweep_variants(variants)


def _effective_sweep_grid(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    grid: Mapping[str, Sequence[object]],
) -> dict[str, list[object]]:
    """过滤单策略无效 detector 参数，避免未启用模块进入 sweep 热路径。"""
    normalized = {str(key): list(values) for key, values in grid.items()}
    if isinstance(config, PortfolioExperimentConfig) and "detectors" in normalized:
        return normalized
    if isinstance(config, SingleStrategyExperimentConfig) and "detector" in normalized:
        return normalized

    active_fields = _active_detector_parameter_fields(config)
    return {
        key: values
        for key, values in normalized.items()
        if key not in ALL_DETECTOR_PARAMETER_FIELDS or key in active_fields
    }


def _deduplicate_sweep_grid_values(values: Sequence[object]) -> list[object]:
    """在笛卡尔积展开前去掉重复参数值，避免重复配置进入热路径。"""
    seen: set[str] = set()
    deduplicated: list[object] = []
    for value in values:
        fingerprint = _sweep_grid_value_fingerprint(value)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduplicated.append(value)
    return deduplicated


def _sweep_grid_value_fingerprint(value: object) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _deduplicate_sweep_variants(
    variants: Sequence[PortfolioExperimentConfig | SingleStrategyExperimentConfig],
) -> list[PortfolioExperimentConfig | SingleStrategyExperimentConfig]:
    """按完整配置指纹去掉重复 case，避免重复 grid 值造成无效回测。"""
    seen: set[str] = set()
    deduplicated: list[PortfolioExperimentConfig | SingleStrategyExperimentConfig] = []
    for variant in variants:
        config_hash = _case_config_hash(variant)
        if config_hash in seen:
            continue
        seen.add(config_hash)
        deduplicated.append(variant)
    return deduplicated


def _sweep_parameter_record(
    base: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    variant: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    keys: Sequence[str],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for key in keys:
        value = getattr(variant, key)
        record[key] = ",".join(value) if isinstance(value, tuple) else value
    for key in ("detectors", "detector", "side_mode", "intrabar_exit_policy"):
        if key not in record:
            if not hasattr(base, key):
                continue
            value = getattr(base, key)
            record[key] = ",".join(value) if isinstance(value, tuple) else value
    return record


def _rank_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    """给参数遍历表生成稳定排名；收益优先，回撤和成交数辅助，最后用 case_name 打破并列。"""
    if "sweep_rank" in table.columns:
        table = table.drop(columns=["sweep_rank"])
    for column in ("pareto_rank", "is_pareto_efficient"):
        if column in table.columns:
            table = table.drop(columns=[column])
    sort_spec = [
        ("total_return", False),
        ("max_drawdown", False),
        ("monthly_worst_return", False),
        ("monthly_return_std", True),
        ("monthly_max_consecutive_losses", True),
        ("monthly_max_recovery_periods", True),
        ("trade_count", False),
        ("case_name", True),
    ]
    sort_columns = [column for column, _ in sort_spec if column in table.columns]
    ascending = [direction for column, direction in sort_spec if column in table.columns]
    ranked = (
        table.sort_values(sort_columns, ascending=ascending, kind="mergesort").reset_index(drop=True)
        if sort_columns
        else table.reset_index(drop=True)
    )
    ranked.insert(0, "sweep_rank", range(1, len(ranked) + 1))
    pareto_rank = _pareto_front_ranks(ranked)
    ranked.insert(1, "pareto_rank", pareto_rank)
    ranked.insert(2, "is_pareto_efficient", [rank == 1 for rank in pareto_rank])
    if "case_config_hash" in ranked.columns:
        values = ranked.pop("case_config_hash")
        ranked.insert(3, "case_config_hash", values)
    return ranked


def _case_config_hash(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> str:
    """给完整实验配置生成稳定指纹，用于 sweep 行跨机器复现和对照。"""
    payload = json.dumps(_case_config_hash_payload(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _case_config_hash_payload(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> dict[str, object]:
    payload = _json_ready(asdict(config))
    for key in NON_REPRODUCIBLE_CONFIG_HASH_FIELDS:
        payload.pop(key, None)
    active_fields = _active_detector_parameter_fields(config)
    for key in ALL_DETECTOR_PARAMETER_FIELDS.difference(active_fields):
        payload.pop(key, None)
    return payload


def _active_detector_parameter_fields(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> set[str]:
    detector_names = config.detectors if isinstance(config, PortfolioExperimentConfig) else (config.detector,)
    active_fields: set[str] = set()
    for detector_name in detector_names:
        active_fields.update(DETECTOR_PARAMETER_FIELDS.get(detector_name, frozenset()))
    if str(config.higher_timeframe).strip():
        active_fields.update(DETECTOR_PARAMETER_FIELDS["trend"])
    return active_fields


def _sweep_case_config_records(result: PortfolioSweepResult | SingleStrategySweepResult) -> list[dict[str, object]]:
    """按 sweep 表排序输出每个 case 的完整配置，便于从结果行直接复现实验。"""
    records_by_hash: dict[str, dict[str, object]] = {}
    for case_index, variant in enumerate(_sweep_variants(result.config, result.grid), start=1):
        config_hash = _case_config_hash(variant)
        records_by_hash[config_hash] = {
            "case_name": f"{result.config.name}-{case_index:03d}",
            "case_config_hash": config_hash,
            "grid_fields": list(result.grid),
            "config": _json_ready(asdict(variant)),
        }
    if result.table.empty or "case_config_hash" not in result.table.columns:
        return list(records_by_hash.values())

    records: list[dict[str, object]] = []
    for row in result.table.to_dict("records"):
        config_hash = str(row["case_config_hash"])
        record = records_by_hash.get(config_hash)
        if record is None:
            raise ValueError(f"sweep 表包含未知 case_config_hash：{config_hash}")
        enriched = dict(record)
        for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
            if column in row:
                enriched[column] = row[column]
        records.append(enriched)
    return records


def load_sweep_case_config(
    path: str | Path,
    *,
    case_config_hash: str = "",
    case_name: str = "",
) -> PortfolioExperimentConfig | SingleStrategyExperimentConfig:
    """从 case_configs.jsonl 读取单个 case 的完整配置，用于精确回放参数遍历结果。"""
    if not case_config_hash and not case_name:
        raise ValueError("必须提供 case_config_hash 或 case_name。")
    records = _read_jsonl(Path(path).expanduser())
    matches = [
        record
        for record in records
        if (not case_config_hash or str(record.get("case_config_hash", "")) == case_config_hash)
        and (not case_name or str(record.get("case_name", "")) == case_name)
    ]
    if not matches:
        raise ValueError("未找到匹配的 sweep case 配置。")
    if len(matches) > 1:
        raise ValueError("匹配到多个 sweep case 配置，请同时指定 case_config_hash 和 case_name。")
    config_payload = matches[0].get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("case 配置缺少 config 对象。")
    config = _experiment_config_from_payload(config_payload)
    recorded_hash = str(matches[0].get("case_config_hash", ""))
    actual_hash = _case_config_hash(config)
    if recorded_hash and recorded_hash != actual_hash:
        raise ValueError("case_config_hash 与 config 内容不一致，拒绝回放被篡改或损坏的 case 配置。")
    return config


def _experiment_config_from_payload(payload: Mapping[str, object]) -> PortfolioExperimentConfig | SingleStrategyExperimentConfig:
    data = dict(payload)
    if "symbols" in data:
        data["symbols"] = tuple(data["symbols"]) if isinstance(data["symbols"], list) else data["symbols"]
    if "detectors" in data:
        data["detectors"] = tuple(data["detectors"]) if isinstance(data["detectors"], list) else data["detectors"]
        return PortfolioExperimentConfig(**data)
    if "detector" in data:
        return SingleStrategyExperimentConfig(**data)
    raise ValueError("case 配置无法识别为单策略或组合实验配置。")


def _pareto_front_ranks(table: pd.DataFrame) -> list[int]:
    """按收益、回撤、Ulcer 和交易样本数给参数组分层；第一层是互不支配的候选集。"""
    if table.empty:
        return []
    scores = _pareto_score_table(table)
    if scores.empty:
        return [1] * len(table)

    values = scores.to_numpy(dtype=float)
    dominates = _pareto_dominance_matrix(values)
    remaining = np.ones(len(values), dtype=bool)
    ranks = [0] * len(values)
    current_rank = 1
    while remaining.any():
        active_dominance = dominates[np.ix_(remaining, remaining)]
        active_indices = np.flatnonzero(remaining)
        front_indices = active_indices[~active_dominance.any(axis=0)]
        if len(front_indices) == 0:
            front_indices = active_indices
        for index in front_indices:
            ranks[int(index)] = current_rank
        remaining[front_indices] = False
        current_rank += 1
    return ranks


def _pareto_dominance_matrix(values: np.ndarray) -> np.ndarray:
    """一次性计算支配关系矩阵；行支配列为 True，对角线永远为 False。"""
    if values.size == 0:
        return np.zeros((len(values), len(values)), dtype=bool)
    greater_or_equal = values[:, None, :] >= values[None, :, :]
    strictly_greater = values[:, None, :] > values[None, :, :]
    dominates = greater_or_equal.all(axis=2) & strictly_greater.any(axis=2)
    np.fill_diagonal(dominates, False)
    return dominates


def _pareto_score_table(table: pd.DataFrame) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for column, direction in SWEEP_PARETO_OBJECTIVES:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        scores[column] = -values if direction == "min" else values
    if not scores:
        return pd.DataFrame(index=table.index)
    return pd.DataFrame(scores, index=table.index).fillna(float("-inf"))


def _grouped_trade_statistics(trades: pd.DataFrame, *, by: str | Sequence[str]) -> pd.DataFrame:
    fields = (by,) if isinstance(by, str) else tuple(by)
    missing = [field for field in fields if field not in trades.columns]
    if missing:
        return pd.DataFrame(columns=pd.Index([*fields, *STAT_KEYS]))
    return compute_grouped_trade_statistics(trades, by=by)


def _strategy_trade_statistics(
    trades: pd.DataFrame,
    strategies: Sequence[object],
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按策略汇总成交表现；保留已启用但没有成交的策略。"""
    columns = pd.Index(["strategy_name", *STAT_KEYS])
    stats = _grouped_trade_statistics(trades, by="strategy_name").reindex(columns=columns)
    strategy_names = _strategy_names_for_statistics(strategies, order_decisions, filter_decisions)
    if not strategy_names:
        return stats
    existing_names = set()
    if not stats.empty and "strategy_name" in stats.columns:
        existing_names = {name for name in stats["strategy_name"].map(_setup_label) if name}
    missing_names = [name for name in strategy_names if name not in existing_names]
    if not missing_names:
        return _sort_strategy_statistics(stats, strategy_names)
    zero_rows = pd.DataFrame(
        [
            {
                "strategy_name": strategy_name,
                **{stat_key: 0.0 for stat_key in STAT_KEYS},
            }
            for strategy_name in missing_names
        ],
        columns=columns,
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_strategy_statistics(pd.concat(frames, ignore_index=True), strategy_names)


def _strategy_names_for_statistics(
    strategies: Sequence[object],
    *decision_frames: pd.DataFrame | None,
) -> tuple[str, ...]:
    """生成策略统计行的稳定顺序，优先使用本次实际执行的策略对象。"""
    names: list[str] = []
    for strategy in strategies:
        name = _setup_label(strategy if isinstance(strategy, str) else getattr(strategy, "name", ""))
        if name and name not in names:
            names.append(name)
    for strategy_name in _strategy_names_from_decisions(*decision_frames):
        if strategy_name not in names:
            names.append(strategy_name)
    return tuple(names)


def _strategy_names_from_decisions(*decision_frames: pd.DataFrame | None) -> tuple[str, ...]:
    names: list[str] = []
    for frame in decision_frames:
        if frame is None or frame.empty or "strategy_name" not in frame.columns:
            continue
        for strategy_name in frame["strategy_name"].map(_setup_label):
            if strategy_name and strategy_name not in names:
                names.append(strategy_name)
    return tuple(names)


def _sort_strategy_statistics(stats: pd.DataFrame, strategy_names: Sequence[str]) -> pd.DataFrame:
    columns = pd.Index(["strategy_name", *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    order = {name: index for index, name in enumerate(strategy_names)}
    result = stats.reindex(columns=columns).copy()
    result["_strategy_label"] = result["strategy_name"].map(_setup_label)
    result["_strategy_order"] = result["_strategy_label"].map(lambda name: order.get(name, len(order)))
    return (
        result.sort_values(["_strategy_order", "_strategy_label"], kind="mergesort")
        .drop(columns=["_strategy_order", "_strategy_label"])
        .reset_index(drop=True)
    )


def _detector_trade_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按识别模块汇总成交表现；保留已启用但没有成交的 detector。"""
    columns = pd.Index(["detector_name", *STAT_KEYS])
    stats = _grouped_trade_statistics(trades, by="detector_name").reindex(columns=columns)
    detector_names = _detector_names_for_statistics(config, order_decisions, filter_decisions)
    if not detector_names:
        return stats
    existing_names = set()
    if not stats.empty and "detector_name" in stats.columns:
        existing_names = {name for name in stats["detector_name"].map(_setup_label) if name}
    missing_names = [name for name in detector_names if name not in existing_names]
    if not missing_names:
        return _sort_detector_statistics(stats, detector_names)
    zero_rows = pd.DataFrame(
        [
            {
                "detector_name": detector_name,
                **{stat_key: 0.0 for stat_key in STAT_KEYS},
            }
            for detector_name in missing_names
        ],
        columns=columns,
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_detector_statistics(pd.concat(frames, ignore_index=True), detector_names)


def _detector_names_for_statistics(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *decision_frames: pd.DataFrame | None,
) -> tuple[str, ...]:
    """生成 detector 统计行的稳定顺序，优先使用实验配置。"""
    configured = config.detectors if isinstance(config, PortfolioExperimentConfig) else (config.detector,)
    names: list[str] = []
    for detector_name in configured:
        name = _setup_label(detector_name)
        if name and name not in names:
            names.append(name)
    for detector_name in _detector_names_from_decisions(*decision_frames):
        if detector_name not in names:
            names.append(detector_name)
    return tuple(names)


def _detector_names_from_decisions(*decision_frames: pd.DataFrame | None) -> tuple[str, ...]:
    names: list[str] = []
    for frame in decision_frames:
        if frame is None or frame.empty or "detector_name" not in frame.columns:
            continue
        for detector_name in frame["detector_name"].map(_setup_label):
            if detector_name and detector_name not in names:
                names.append(detector_name)
    return tuple(names)


def _sort_detector_statistics(stats: pd.DataFrame, detector_names: Sequence[str]) -> pd.DataFrame:
    columns = pd.Index(["detector_name", *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    order = {name: index for index, name in enumerate(detector_names)}
    result = stats.reindex(columns=columns).copy()
    result["_detector_label"] = result["detector_name"].map(_setup_label)
    result["_detector_order"] = result["_detector_label"].map(lambda name: order.get(name, len(order)))
    return (
        result.sort_values(["_detector_order", "_detector_label"], kind="mergesort")
        .drop(columns=["_detector_order", "_detector_label"])
        .reset_index(drop=True)
    )


def _setup_trade_statistics(
    trades: pd.DataFrame,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 setup 汇总成交表现；保留只有信号或拒单、没有成交的 setup。"""
    stats = _grouped_trade_statistics(trades, by=SETUP_STAT_FIELDS).reindex(
        columns=pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS])
    )
    setup_keys = _setup_keys_from_decisions(order_decisions, filter_decisions)
    if setup_keys.empty:
        return stats
    existing_keys = set(_setup_key_tuples(stats))
    missing_keys = [
        tuple(row)
        for row in setup_keys.loc[:, SETUP_STAT_FIELDS].itertuples(index=False, name=None)
        if tuple(row) not in existing_keys
    ]
    if not missing_keys:
        return _sort_setup_statistics(stats)
    zero_rows = pd.DataFrame(
        [
            {
                **dict(zip(SETUP_STAT_FIELDS, key, strict=True)),
                **{stat_key: 0.0 for stat_key in STAT_KEYS},
            }
            for key in missing_keys
        ],
        columns=pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS]),
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS]))
    return _sort_setup_statistics(pd.concat(frames, ignore_index=True))


def _setup_keys_from_decisions(*decision_frames: pd.DataFrame | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in decision_frames:
        if frame is None or frame.empty or not set(SETUP_STAT_FIELDS).issubset(frame.columns):
            continue
        setup = frame.loc[:, SETUP_STAT_FIELDS].copy()
        for setup_field in SETUP_STAT_FIELDS:
            setup[setup_field] = setup[setup_field].map(_setup_label)
        present = setup.loc[:, SETUP_STAT_FIELDS].ne("").all(axis=1)
        if bool(present.any()):
            frames.append(setup.loc[present, SETUP_STAT_FIELDS])
    if not frames:
        return pd.DataFrame(columns=pd.Index(SETUP_STAT_FIELDS))
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(list(SETUP_STAT_FIELDS), kind="mergesort")
        .reset_index(drop=True)
    )


def _setup_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _setup_key_tuples(stats: pd.DataFrame) -> list[tuple[str, str, str]]:
    if stats.empty or not set(SETUP_STAT_FIELDS).issubset(stats.columns):
        return []
    normalized = stats.loc[:, SETUP_STAT_FIELDS].copy()
    for setup_field in SETUP_STAT_FIELDS:
        normalized[setup_field] = normalized[setup_field].map(_setup_label)
    return [tuple(row) for row in normalized.itertuples(index=False, name=None)]


def _sort_setup_statistics(stats: pd.DataFrame) -> pd.DataFrame:
    columns = pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    return (
        stats.reindex(columns=columns)
        .sort_values(list(SETUP_STAT_FIELDS), kind="mergesort")
        .reset_index(drop=True)
    )


def _trade_dated_equity_curve(backtest: BacktestResult) -> pd.DataFrame:
    equity = backtest.equity_curve.copy()
    if "date" in equity.columns:
        return equity
    trades = backtest.trades.copy()
    if equity.empty or trades.empty or "exit_date" not in trades.columns:
        return equity
    if "trade_no" not in equity.columns:
        return equity
    dated = equity.merge(
        trades.assign(trade_no=range(1, len(trades) + 1))[["trade_no", "exit_date"]],
        on="trade_no",
        how="left",
    )
    if 0 in set(pd.to_numeric(dated["trade_no"], errors="coerce").dropna().astype(int)):
        first_exit_date = pd.to_datetime(trades["exit_date"], errors="coerce").dropna().min()
        if pd.notna(first_exit_date):
            dated.loc[dated["trade_no"].eq(0), "exit_date"] = first_exit_date
    return dated.rename(columns={"exit_date": "date"})


def _json_ready(value):
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    lines = [
        json.dumps(_json_ready(record), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"case 配置文件不存在：{path}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"case 配置第 {line_number} 行不是 JSON 对象。")
        records.append(payload)
    return records
