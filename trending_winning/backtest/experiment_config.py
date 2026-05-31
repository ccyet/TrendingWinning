from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd

from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig
from trending_winning.backtest.models import BacktestConfig
from trending_winning.backtest.portfolio_models import PortfolioConfig
from trending_winning.strategies.multitimeframe import HigherTimeframeAlignmentStrategy, TimeframeAlignmentConfig
from trending_winning.strategies.suite import StrategySuiteConfig


def wrap_higher_timeframe_strategies(
    strategies: Sequence[object],
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    higher_bars: pd.DataFrame,
    *,
    context_fn: Callable[[pd.DataFrame, PortfolioExperimentConfig | SingleStrategyExperimentConfig], pd.DataFrame] | None = None,
) -> list[object]:
    """按实验配置给策略套上大周期方向过滤；无大周期时直接返回原策略。"""
    higher_timeframe = str(config.higher_timeframe).strip()
    if not higher_timeframe:
        return list(strategies)
    build_context = context_fn or higher_timeframe_context
    context = build_context(higher_bars, config)
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


def higher_timeframe_context(
    bars: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> pd.DataFrame:
    """从大周期 K 线生成趋势状态上下文；趋势 detector 延迟导入，避免模块加载时耦合。"""
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


def backtest_config(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> BacktestConfig:
    """把实验配置收敛成撮合层参数，避免 runner 与撮合模型字段耦合。"""
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


def portfolio_config(config: PortfolioExperimentConfig) -> PortfolioConfig:
    """把组合实验配置收敛成资金分配参数，单策略回测不依赖该对象。"""
    return PortfolioConfig(
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
    )


def candidate_cache_key(config: PortfolioExperimentConfig) -> tuple[object, ...]:
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


def order_cache_key(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    suite_config: StrategySuiteConfig,
) -> tuple[object, ...]:
    """订单缓存依赖 detector 参数和策略层门控参数；组合资金参数变化可复用。"""
    higher_timeframe = str(config.higher_timeframe).strip()
    return (
        active_strategy_suite_cache_key(suite_config, include_trend_context=bool(higher_timeframe)),
        higher_timeframe,
        None if not higher_timeframe else config.higher_timeframe_max_age_minutes,
    )


def active_strategy_suite_cache_key(
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
        parts.append((detector_name, detector_cache_parameters(detector_name, suite_config)))
    if include_trend_context and "trend" not in enabled:
        parts.append(("higher_trend_context", detector_cache_parameters("trend", suite_config)))
    return tuple(parts)


def detector_cache_parameters(detector_name: str, suite_config: StrategySuiteConfig) -> tuple[object, ...]:
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


def strategy_suite_config(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> StrategySuiteConfig:
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
