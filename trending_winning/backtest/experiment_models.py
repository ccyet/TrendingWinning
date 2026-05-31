from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trending_winning.backtest.models import BacktestResult


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
    terminal_false_breakout_enabled: bool = False
    terminal_false_breakout_detectors: tuple[str, ...] = ("trend", "channel")
    terminal_false_breakout_lookback: int = 40
    terminal_false_breakout_atr_period: int = 14
    terminal_false_breakout_min_regime_bars: int = 18
    terminal_false_breakout_extension_atr_multiple: float = 2.0
    terminal_false_breakout_edge_lookback: int = 8
    terminal_false_breakout_edge_pos: float = 0.90
    terminal_false_breakout_edge_min_count: int = 3
    terminal_false_breakout_weak_progress_atr: float = 0.35
    terminal_false_breakout_wick_ratio: float = 0.35
    terminal_false_breakout_min_score: int = 3
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
    terminal_false_breakout_enabled: bool = False
    terminal_false_breakout_detectors: tuple[str, ...] = ("trend", "channel")
    terminal_false_breakout_lookback: int = 40
    terminal_false_breakout_atr_period: int = 14
    terminal_false_breakout_min_regime_bars: int = 18
    terminal_false_breakout_extension_atr_multiple: float = 2.0
    terminal_false_breakout_edge_lookback: int = 8
    terminal_false_breakout_edge_pos: float = 0.90
    terminal_false_breakout_edge_min_count: int = 3
    terminal_false_breakout_weak_progress_atr: float = 0.35
    terminal_false_breakout_wick_ratio: float = 0.35
    terminal_false_breakout_min_score: int = 3
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
    signal_lifecycle_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_path_distribution_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostic_report: pd.DataFrame = field(default_factory=pd.DataFrame)
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
    signal_lifecycle_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    detector_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    setup_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_type_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_path_distribution_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostic_report: pd.DataFrame = field(default_factory=pd.DataFrame)
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
