from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter

import pandas as pd

from trending_winning.backtest.engine import BacktestConfig, BacktestResult, run_order_backtest, run_single_strategy_backtest
from trending_winning.backtest.portfolio import (
    PortfolioConfig,
    PortfolioCandidateSet,
    collect_strategy_orders_from_normalized,
    prepare_portfolio_candidates_from_normalized,
    run_portfolio_candidate_backtest_from_normalized,
    run_portfolio_backtest,
)
from trending_winning.backtest.stats import (
    STAT_KEYS,
    compute_decision_reason_statistics,
    compute_grouped_trade_statistics,
    compute_period_returns,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.data.repository import MarketDataRepository
from trending_winning.strategies.diagnostics import collect_strategy_filter_decisions
from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy, TimeframeAlignmentConfig
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
    intrabar_exit_policy: str = "conservative"
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    initial_equity: float = 1.0
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
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    order_decision_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    strategy_filter_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    limit_filter_audit: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class _LoadedExperimentData:
    """实验装载后的标准数据包；集中携带主周期、高周期和审计结果。"""

    bars: pd.DataFrame
    higher_bars: pd.DataFrame
    data_audit: pd.DataFrame
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
    backtest = run_single_strategy_backtest(
        data.bars,
        strategy,
        _backtest_config(config),
        timeframe=config.timeframe,
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=backtest,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
        data_coverage=data.data_audit,
        limit_filter_audit=data.limit_filter_audit,
        strategy_stats=_grouped_trade_statistics(backtest.trades, by="strategy_name"),
        symbol_stats=_grouped_trade_statistics(backtest.trades, by="stock_code"),
        side_stats=_grouped_trade_statistics(backtest.trades, by="side"),
        exit_reason_stats=_grouped_trade_statistics(backtest.trades, by="exit_reason"),
        event_type_stats=_grouped_trade_statistics(backtest.trades, by="event_type"),
        order_decision_stats=compute_decision_reason_statistics(backtest.order_decisions),
        strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("strategy_name", "filter_name", "context_timeframe"),
        ),
        monthly_returns=compute_period_returns(_trade_dated_equity_curve(backtest), freq="M"),
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
    backtest = run_portfolio_backtest(
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
    result = PortfolioExperimentResult(
        config=config,
        backtest=backtest,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        data_coverage=data.data_audit,
        limit_filter_audit=data.limit_filter_audit,
        strategy_stats=_grouped_trade_statistics(backtest.trades, by="strategy_name"),
        symbol_stats=_grouped_trade_statistics(backtest.trades, by="stock_code"),
        side_stats=_grouped_trade_statistics(backtest.trades, by="side"),
        exit_reason_stats=_grouped_trade_statistics(backtest.trades, by="exit_reason"),
        event_type_stats=_grouped_trade_statistics(backtest.trades, by="event_type"),
        order_decision_stats=compute_decision_reason_statistics(backtest.order_decisions),
        strategy_filter_stats=compute_decision_reason_statistics(
            backtest.strategy_filter_decisions,
            group_fields=("strategy_name", "filter_name", "context_timeframe"),
        ),
        monthly_returns=compute_period_returns(backtest.equity_curve, freq="M"),
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
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)

    rows: list[dict[str, object]] = []
    orders_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
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
            orders = collect_strategy_orders_from_normalized(
                data.bars,
                strategies,
                timeframe=variant.timeframe,
            )
            orders_by_config[order_key] = orders
            filter_decisions_by_config[order_key] = collect_strategy_filter_decisions(strategies)
        backtest_config = _backtest_config(variant)
        candidate_key = (order_key, _candidate_cache_key(variant))
        candidate_set = candidates_by_execution.get(candidate_key)
        candidate_cache_status = "hit"
        if candidate_set is None:
            candidate_cache_status = "miss"
            candidate_set = prepare_portfolio_candidates_from_normalized(data.bars, orders, backtest_config)
            candidates_by_execution[candidate_key] = candidate_set
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
        row: dict[str, object] = {
            "case_name": f"{config.name}-{case_index:03d}",
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
        row.update(_sweep_parameter_record(config, variant, grid.keys()))
        row.update(summarize_order_decisions(backtest.order_decisions))
        row.update(backtest.stats)
        row.update(summarize_strategy_filter_decisions(filter_decisions_by_config.get(order_key, pd.DataFrame())))
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["total_return", "max_drawdown", "trade_count"], ascending=[False, False, False]).reset_index(
            drop=True
        )
    result = PortfolioSweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
        limit_filter_audit=data.limit_filter_audit,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
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
    repo = MarketDataRepository(config.data_root, adjust=config.adjust)
    data = _load_experiment_data(repo, config)

    rows: list[dict[str, object]] = []
    orders_by_config: dict[tuple[object, ...], pd.DataFrame] = {}
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
            orders = strategy.generate_orders(data.bars, timeframe=variant.timeframe)
            orders_by_config[order_key] = orders
            filter_decisions_by_config[order_key] = collect_strategy_filter_decisions([strategy])
        filter_decisions = filter_decisions_by_config.get(order_key, pd.DataFrame())
        backtest = _with_strategy_filter_decisions(
            run_order_backtest(data.bars, orders, _backtest_config(variant)),
            filter_decisions,
        )
        case_elapsed = max(perf_counter() - case_start, 1e-12)
        row: dict[str, object] = {
            "case_name": f"{config.name}-{case_index:03d}",
            "bar_count": int(len(data.bars)),
            "trade_count": int(len(backtest.trades)),
            "equity_points": int(len(backtest.equity_curve)),
            "elapsed_seconds": float(case_elapsed),
            "bars_per_second": float(len(data.bars) / case_elapsed),
            "order_cache_status": order_cache_status,
            "generated_order_count": int(len(orders)),
        }
        row.update(_sweep_parameter_record(config, variant, grid.keys()))
        row.update(summarize_order_decisions(backtest.order_decisions))
        row.update(backtest.stats)
        row.update(summarize_strategy_filter_decisions(filter_decisions))
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["total_return", "max_drawdown", "trade_count"], ascending=[False, False, False]).reset_index(
            drop=True
        )
    result = SingleStrategySweepResult(
        config=config,
        grid={key: list(values) for key, values in grid.items()},
        table=table,
        data_coverage=data.data_audit,
        limit_filter_audit=data.limit_filter_audit,
        input_bar_count=int(len(data.bars)),
        filtered_limit_open_count=int(len(data.filtered_limit_open_days)),
        elapsed_seconds=float(max(perf_counter() - start_time, 1e-12)),
    )
    if save:
        save_single_strategy_sweep(result)
    return result


def _load_experiment_data(
    repo: MarketDataRepository,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> _LoadedExperimentData:
    higher_timeframe = str(config.higher_timeframe).strip()
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
        limit_filter_audit=bundle.limit_filter_audit,
        filtered_limit_open_days=bundle.filtered_limit_open_days,
    )


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
    )


def _candidate_cache_key(config: PortfolioExperimentConfig) -> tuple[object, ...]:
    """候选成交只依赖撮合路径参数；组合分配和初始资金变化可复用。"""
    return (
        int(config.max_holding_bars),
        float(config.fee_rate),
        float(config.slippage_bps),
        str(config.intrabar_exit_policy),
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


def save_single_strategy_experiment(result: SingleStrategyExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(_json_dump(_json_ready(asdict(result.config))))
    stats = dict(result.backtest.stats)
    stats["filtered_limit_open_count"] = float(result.filtered_limit_open_count)
    stats["elapsed_seconds"] = float(result.elapsed_seconds)
    (output_dir / "stats.json").write_text(_json_dump(_json_ready(stats)))
    result.backtest.trades.to_csv(output_dir / "trades.csv", index=False)
    result.backtest.order_decisions.to_csv(output_dir / "order_decisions.csv", index=False)
    result.backtest.strategy_filter_decisions.to_csv(output_dir / "strategy_filter_decisions.csv", index=False)
    result.backtest.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    return output_dir


def save_portfolio_experiment(result: PortfolioExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(_json_dump(_json_ready(asdict(result.config))))
    stats = dict(result.backtest.stats)
    stats["filtered_limit_open_count"] = float(result.filtered_limit_open_count)
    (output_dir / "stats.json").write_text(_json_dump(_json_ready(stats)))
    result.backtest.trades.to_csv(output_dir / "trades.csv", index=False)
    result.backtest.order_decisions.to_csv(output_dir / "order_decisions.csv", index=False)
    result.backtest.strategy_filter_decisions.to_csv(output_dir / "strategy_filter_decisions.csv", index=False)
    result.backtest.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
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
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    return output_dir


def save_single_strategy_sweep(result: SingleStrategySweepResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_payload = _json_ready(asdict(result.config))
    config_payload["sweep_grid"] = _json_ready(result.grid)
    (output_dir / "config.json").write_text(_json_dump(config_payload))
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    return output_dir


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
    keys = list(grid)
    value_lists = [list(grid[key]) for key in keys]
    empty_keys = [key for key, values in zip(keys, value_lists, strict=False) if not values]
    if empty_keys:
        raise ValueError(f"grid 字段不能为空：{', '.join(empty_keys)}")
    return [
        replace(config, **dict(zip(keys, values, strict=False)))
        for values in product(*value_lists)
    ]


def _sweep_parameter_record(
    base: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    variant: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    keys: Sequence[str],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for key in keys:
        value = getattr(variant, key)
        record[key] = ",".join(value) if isinstance(value, tuple) else value
    for key in ("detectors", "detector", "intrabar_exit_policy"):
        if key not in record:
            if not hasattr(base, key):
                continue
            value = getattr(base, key)
            record[key] = ",".join(value) if isinstance(value, tuple) else value
    return record


def _grouped_trade_statistics(trades: pd.DataFrame, *, by: str) -> pd.DataFrame:
    if by not in trades.columns:
        return pd.DataFrame(columns=pd.Index([by, *STAT_KEYS]))
    return compute_grouped_trade_statistics(trades, by=by)


def _trade_dated_equity_curve(backtest: BacktestResult) -> pd.DataFrame:
    equity = backtest.equity_curve.copy()
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
