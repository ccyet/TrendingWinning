from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig
from trending_winning.data.audit import DATA_GAP_EPISODE_COLUMNS
from trending_winning.backtest.models import BacktestResult
from trending_winning.data.summary import summarize_data_management


@dataclass(frozen=True)
class LoadedExperimentData:
    """实验装载后的标准数据包；集中携带主周期、高周期和数据审计结果。"""

    bars: pd.DataFrame
    higher_bars: pd.DataFrame
    data_audit: pd.DataFrame
    data_gap_episodes: pd.DataFrame
    data_inventory: pd.DataFrame
    limit_filter_audit: pd.DataFrame
    filtered_limit_open_days: pd.DataFrame


def load_experiment_data(
    repo: Any,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> LoadedExperimentData:
    """按实验配置装载主周期、高周期、库存和过滤审计，供单策略/组合回测共用。"""
    higher_timeframe = str(config.higher_timeframe).strip()
    data_inventory = repo.inventory(
        timeframes=tuple(experiment_inventory_timeframes(config)),
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
        return LoadedExperimentData(
            bars=bundle.bars_by_timeframe.get(config.timeframe, pd.DataFrame()),
            higher_bars=bundle.bars_by_timeframe.get(higher_timeframe, pd.DataFrame()),
            data_audit=bundle.data_audit,
            data_gap_episodes=_bundle_data_gap_episodes(bundle),
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
    return LoadedExperimentData(
        bars=bundle.bars,
        higher_bars=pd.DataFrame(),
        data_audit=bundle.data_audit,
        data_gap_episodes=_bundle_data_gap_episodes(bundle),
        data_inventory=data_inventory,
        limit_filter_audit=bundle.limit_filter_audit,
        filtered_limit_open_days=bundle.filtered_limit_open_days,
    )


def _bundle_data_gap_episodes(bundle: Any) -> pd.DataFrame:
    return getattr(bundle, "data_gap_episodes", pd.DataFrame(columns=DATA_GAP_EPISODE_COLUMNS))


def experiment_inventory_timeframes(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> list[str]:
    """实验结果需要保存日 K 依赖、主周期和可选高周期的本地缓存快照。"""
    timeframes = ["1d", str(config.timeframe)]
    higher_timeframe = str(config.higher_timeframe).strip()
    if higher_timeframe and higher_timeframe not in timeframes:
        timeframes.append(higher_timeframe)
    return timeframes


def with_data_management_statistics(
    result: BacktestResult,
    data: LoadedExperimentData,
    *,
    min_coverage_ratio: float | None,
) -> BacktestResult:
    """运行态结果携带数据审计摘要，保证 Web/CLI 和保存产物统计口径一致。"""
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
