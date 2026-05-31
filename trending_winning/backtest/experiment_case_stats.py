from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig
from trending_winning.backtest.experiment_output import symbol_metadata_for_config
from trending_winning.backtest.reporting import (
    SETUP_STAT_FIELDS,
    detector_trade_statistics,
    grouped_trade_statistics,
    setup_trade_statistics,
    strategy_trade_statistics,
)
from trending_winning.backtest.stats import STAT_KEYS, compute_decision_reason_statistics
from trending_winning.data.schema import unique_symbols

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


def symbol_grouped_trade_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *,
    symbol_name_by_code: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """标的维度统计优先带股票名称；无成交股票保留零值行，方便回测复核。"""
    stats = grouped_trade_statistics(trades, by="stock_code")
    if "stock_name" in stats.columns:
        return stats
    if "stock_code" not in stats.columns:
        return stats
    symbols = unique_symbols(tuple(config.symbols))
    name_by_symbol = dict(symbol_name_by_code) if symbol_name_by_code is not None else symbol_name_map_for_config(config)
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


def symbol_name_map_for_config(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
) -> dict[str, str]:
    metadata = symbol_metadata_for_config(config)
    return {str(row.stock_code): str(row.stock_name) for row in metadata.itertuples(index=False)}


def configured_detector_names(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> tuple[str, ...]:
    return tuple(config.detectors) if isinstance(config, PortfolioExperimentConfig) else (config.detector,)


def case_setup_statistics(
    trades: pd.DataFrame,
    *,
    case_name: str,
    case_config_hash: str,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 case 汇总 setup 表现；没有成交但出现过信号的 setup 也保留零行。"""
    stats = setup_trade_statistics(trades, order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=pd.Index(["case_name", "case_config_hash", *SETUP_STAT_FIELDS, *STAT_KEYS]))
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats


def case_strategy_statistics(
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
    stats = strategy_trade_statistics(trades, strategies, order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=columns)
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats.reindex(columns=columns)


def case_detector_statistics(
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
    stats = detector_trade_statistics(trades, configured_detector_names(config), order_decisions, filter_decisions)
    if stats.empty:
        return pd.DataFrame(columns=columns)
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats.reindex(columns=columns)


def case_symbol_statistics(
    trades: pd.DataFrame,
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    *,
    symbol_name_by_code: Mapping[str, str],
    case_name: str,
    case_config_hash: str,
) -> pd.DataFrame:
    """按 case 汇总标的表现；没有成交的样本股票也保留零值行。"""
    stats = symbol_grouped_trade_statistics(
        trades,
        config,
        symbol_name_by_code=symbol_name_by_code,
    )
    if stats.empty:
        return pd.DataFrame(columns=pd.Index(["case_name", "case_config_hash", "stock_name", "stock_code", *STAT_KEYS]))
    stats.insert(0, "case_config_hash", case_config_hash)
    stats.insert(0, "case_name", case_name)
    return stats


def case_decision_statistics(
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


def concat_case_strategy_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "strategy_name", *STAT_KEYS])
    return _concat_frames(frames, columns)


def concat_case_detector_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "detector_name", *STAT_KEYS])
    return _concat_frames(frames, columns)


def concat_case_setup_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", *SETUP_STAT_FIELDS, *STAT_KEYS])
    return _concat_frames(frames, columns)


def concat_case_symbol_statistics(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    columns = pd.Index(["case_name", "case_config_hash", "stock_name", "stock_code", *STAT_KEYS])
    return _concat_frames(frames, columns)


def concat_case_decision_statistics(frames: Sequence[pd.DataFrame], *, group_fields: tuple[str, ...]) -> pd.DataFrame:
    columns = _case_decision_columns(group_fields)
    return _concat_frames(frames, columns)


def ranked_case_strategy_statistics(case_strategy: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    ranked = _ranked_case_statistics(
        case_strategy,
        table,
        columns=pd.Index(SWEEP_CASE_STRATEGY_COLUMNS),
        sort_columns=["sweep_rank", "case_name", "strategy_name"],
    )
    return ranked


def ranked_case_detector_statistics(case_detector: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    return _ranked_case_statistics(
        case_detector,
        table,
        columns=pd.Index(SWEEP_CASE_DETECTOR_COLUMNS),
        sort_columns=["sweep_rank", "case_name", "detector_name"],
    )


def ranked_case_setup_statistics(case_setup: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    return _ranked_case_statistics(
        case_setup,
        table,
        columns=pd.Index(SWEEP_CASE_SETUP_COLUMNS),
        sort_columns=["sweep_rank", "case_name", *SETUP_STAT_FIELDS],
    )


def ranked_case_symbol_statistics(case_symbol: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    return _ranked_case_statistics(
        case_symbol,
        table,
        columns=pd.Index(SWEEP_CASE_SYMBOL_COLUMNS),
        sort_columns=["sweep_rank", "case_name", "stock_name", "stock_code"],
    )


def ranked_case_decision_statistics(
    case_decisions: pd.DataFrame,
    table: pd.DataFrame,
    *,
    group_fields: tuple[str, ...],
) -> pd.DataFrame:
    columns = _ranked_case_decision_columns(group_fields)
    sort_columns = ["sweep_rank", "case_name", *group_fields, "status", "reason"]
    return _ranked_case_statistics(case_decisions, table, columns=columns, sort_columns=sort_columns)


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


def _concat_frames(frames: Sequence[pd.DataFrame], columns: pd.Index) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=columns)
    return pd.concat(non_empty, ignore_index=True).reindex(columns=columns)


def _ranked_case_statistics(
    case_stats: pd.DataFrame,
    table: pd.DataFrame,
    *,
    columns: pd.Index,
    sort_columns: Sequence[str],
) -> pd.DataFrame:
    if case_stats.empty:
        return pd.DataFrame(columns=columns)
    rank_columns = ["case_config_hash", "sweep_rank", "pareto_rank", "is_pareto_efficient"]
    ranks = table.loc[:, [column for column in rank_columns if column in table.columns]].copy()
    merged = case_stats.merge(ranks, on="case_config_hash", how="left")
    for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged.reindex(columns=columns).sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)


def _case_decision_columns(group_fields: tuple[str, ...]) -> pd.Index:
    stats_columns = compute_decision_reason_statistics(pd.DataFrame(), group_fields=group_fields).columns
    return pd.Index(["case_name", "case_config_hash", *stats_columns])


def _ranked_case_decision_columns(group_fields: tuple[str, ...]) -> pd.Index:
    return pd.Index(["sweep_rank", "pareto_rank", "is_pareto_efficient", *_case_decision_columns(group_fields)])
