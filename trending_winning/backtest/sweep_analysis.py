from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math

import numpy as np
import pandas as pd


SWEEP_PARETO_OBJECTIVES = (
    ("total_return", "max"),
    ("max_drawdown", "max"),
    ("ulcer_index", "min"),
    ("monthly_worst_return", "max"),
    ("monthly_return_std", "min"),
    ("trade_count", "max"),
)

PARAMETER_SUMMARY_METRICS = (
    ("risk_adjusted_score", "avg_risk_adjusted_score", "mean"),
    ("risk_adjusted_score", "median_risk_adjusted_score", "median"),
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


def rank_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    """给参数遍历表生成稳定排名，并附加 Pareto 分层。"""
    if "sweep_rank" in table.columns:
        table = table.drop(columns=["sweep_rank"])
    for column in ("pareto_rank", "is_pareto_efficient", "risk_adjusted_rank", "risk_adjusted_score"):
        if column in table.columns:
            table = table.drop(columns=[column])
    table = table.assign(risk_adjusted_score=risk_adjusted_scores(table))
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
    pareto_rank = pareto_front_ranks(ranked)
    ranked.insert(1, "pareto_rank", pareto_rank)
    ranked.insert(2, "is_pareto_efficient", [rank == 1 for rank in pareto_rank])
    has_case_hash = "case_config_hash" in ranked.columns
    if has_case_hash:
        values = ranked.pop("case_config_hash")
        ranked.insert(3, "case_config_hash", values)
    score_values = ranked.pop("risk_adjusted_score")
    risk_insert_at = 4 if has_case_hash else 3
    ranked.insert(risk_insert_at, "risk_adjusted_rank", risk_adjusted_ranks(ranked, score_values))
    ranked.insert(risk_insert_at + 1, "risk_adjusted_score", score_values)
    return ranked


def pareto_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    """提取第一层 Pareto 候选，保持 sweep.csv 的列和排名顺序。"""
    if table.empty or "pareto_rank" not in table.columns:
        return table.iloc[0:0].copy()
    ranks = pd.to_numeric(table["pareto_rank"], errors="coerce")
    return table.loc[ranks.eq(1)].copy()


def parameter_summary_table(table: pd.DataFrame, grid: Mapping[str, Sequence[object]]) -> pd.DataFrame:
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
    if table.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for parameter in grid:
        if parameter not in table.columns:
            continue
        labels = table[parameter].map(parameter_value_label)
        for value, group in table.groupby(labels, sort=False, dropna=False):
            best = group.sort_values("sweep_rank", ascending=True, kind="mergesort").iloc[0]
            case_count = int(len(group))
            pareto_case_count = truthy_column_count(group, "is_pareto_efficient")
            row: dict[str, object] = {
                "parameter": str(parameter),
                "value": str(value),
                "case_count": case_count,
                "pareto_case_count": pareto_case_count,
                **parameter_robustness_metrics(group, case_count=case_count, pareto_case_count=pareto_case_count),
                "best_sweep_rank": json_scalar(best.get("sweep_rank")),
                "best_case_name": str(best.get("case_name", "")),
                "best_case_config_hash": str(best.get("case_config_hash", "")),
            }
            for source, output, method in PARAMETER_SUMMARY_METRICS:
                row[output] = numeric_group_metric(group, source, method)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=columns)
    summary = pd.DataFrame(rows, columns=columns)
    return summary.sort_values(["parameter", "best_sweep_rank", "value"], kind="mergesort").reset_index(drop=True)


def parameter_robustness_metrics(
    group: pd.DataFrame,
    *,
    case_count: int,
    pareto_case_count: int,
) -> dict[str, float]:
    """汇总单个参数值的稳健性，避免只按平均收益选参数。"""
    total_return = numeric_group_values(group, "total_return")
    positive_return_count = int(total_return.gt(0).sum()) if not total_return.empty else 0
    return {
        "pareto_hit_rate": float(pareto_case_count / case_count) if case_count else 0.0,
        "positive_return_case_count": float(positive_return_count),
        "positive_return_rate": float(positive_return_count / case_count) if case_count else 0.0,
        "std_total_return": float(total_return.std(ddof=0)) if not total_return.empty else 0.0,
        "best_total_return": float(total_return.max()) if not total_return.empty else 0.0,
        "worst_total_return": float(total_return.min()) if not total_return.empty else 0.0,
    }


def pareto_front_ranks(
    table: pd.DataFrame,
    *,
    dominance_matrix_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[int]:
    """按收益、回撤、Ulcer、月度稳定性和交易样本数给参数组分层。"""
    if table.empty:
        return []
    scores = pareto_score_table(table)
    if scores.empty:
        return [1] * len(table)

    values = scores.to_numpy(dtype=float)
    dominance_fn = dominance_matrix_fn or pareto_dominance_matrix
    dominates = dominance_fn(values)
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


def pareto_dominance_matrix(values: np.ndarray) -> np.ndarray:
    """一次性计算支配关系矩阵；行支配列为 True，对角线永远为 False。"""
    if values.size == 0:
        return np.zeros((len(values), len(values)), dtype=bool)
    greater_or_equal = values[:, None, :] >= values[None, :, :]
    strictly_greater = values[:, None, :] > values[None, :, :]
    dominates = greater_or_equal.all(axis=2) & strictly_greater.any(axis=2)
    np.fill_diagonal(dominates, False)
    return dominates


def pareto_score_table(table: pd.DataFrame) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for column, direction in SWEEP_PARETO_OBJECTIVES:
        if column not in table.columns:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        scores[column] = -values if direction == "min" else values
    if not scores:
        return pd.DataFrame(index=table.index)
    return pd.DataFrame(scores, index=table.index).fillna(float("-inf"))


def risk_adjusted_scores(table: pd.DataFrame) -> pd.Series:
    """把收益、回撤、稳定性、路径效率、样本量和诊断状态压成 0-100 风险质量分。"""
    if table.empty:
        return pd.Series(dtype=float, index=table.index)

    component_weights = {
        "total_return": 0.30,
        "max_drawdown": 0.22,
        "monthly_worst_return": 0.15,
        "monthly_return_std": 0.10,
        "ulcer_index": 0.08,
        "return_per_exposure_bar": 0.08,
        "trade_count": 0.07,
    }
    components = pd.Series(0.0, index=table.index, dtype=float)
    for column, weight in component_weights.items():
        components += weight * _risk_score_component(table, column)

    penalty = _diagnostic_risk_penalty(table)
    return (components.mul(100.0) - penalty).clip(lower=0.0, upper=100.0).round(6)


def risk_adjusted_ranks(table: pd.DataFrame, score: pd.Series) -> list[int]:
    """风险评分的独立排名；不替换 sweep_rank，供用户按稳健性筛选。"""
    if table.empty:
        return []
    sortable = pd.DataFrame(
        {
            "_score": pd.to_numeric(score, errors="coerce").fillna(0.0),
            "_case_name": table.get("case_name", pd.Series([""] * len(table), index=table.index)).astype(str),
            "_position": range(len(table)),
        },
        index=table.index,
    )
    ordered = sortable.sort_values(["_score", "_case_name"], ascending=[False, True], kind="mergesort")
    ranks = pd.Series(range(1, len(ordered) + 1), index=ordered.index)
    return ranks.loc[table.index].astype(int).tolist()


def _risk_score_component(table: pd.DataFrame, column: str) -> pd.Series:
    if column == "trade_count":
        return _sample_size_score(table)
    if column not in table.columns:
        return pd.Series(0.5, index=table.index, dtype=float)
    values = pd.to_numeric(table[column], errors="coerce")
    if column in {"monthly_return_std", "ulcer_index"}:
        values = -values
    return _percentile_score(values)


def _sample_size_score(table: pd.DataFrame) -> pd.Series:
    if "trade_count" not in table.columns:
        return pd.Series(0.5, index=table.index, dtype=float)
    values = pd.to_numeric(table["trade_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    max_value = float(values.max()) if not values.empty else 0.0
    if max_value <= 0:
        return pd.Series(0.0, index=table.index, dtype=float)
    return np.log1p(values) / math.log1p(max_value)


def _percentile_score(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(0.5, index=values.index, dtype=float)
    if len(valid) == 1 or float(valid.max()) == float(valid.min()):
        score = pd.Series(0.5, index=values.index, dtype=float)
        score.loc[valid.index] = 1.0
        return score
    score = values.rank(method="average", pct=True)
    return score.fillna(0.5).astype(float)


def _diagnostic_risk_penalty(table: pd.DataFrame) -> pd.Series:
    failed = _optional_numeric_column(table, "diagnostic_failed_count")
    attention = _optional_numeric_column(table, "diagnostic_attention_count")
    severity = _optional_numeric_column(table, "diagnostic_max_severity")
    return (failed.mul(15.0) + attention.mul(6.0) + severity.mul(8.0)).clip(lower=0.0, upper=45.0)


def _optional_numeric_column(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(0.0, index=table.index, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").fillna(0.0).astype(float)


def numeric_group_metric(group: pd.DataFrame, column: str, method: str) -> float:
    values = numeric_group_values(group, column)
    if values.empty:
        return 0.0
    if method == "median":
        return float(values.median())
    return float(values.mean())


def numeric_group_values(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(group[column], errors="coerce").dropna()


def parameter_value_label(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, dict | list | tuple):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if pd.isna(value):
        return ""
    return str(value)


def truthy_column_count(table: pd.DataFrame, column: str) -> int:
    if table.empty or column not in table.columns:
        return 0
    values = table[column]
    if pd.api.types.is_bool_dtype(values):
        return int(values.fillna(False).sum())
    normalized = values.fillna(False).astype(str).str.lower()
    return int(normalized.isin(("true", "1", "yes")).sum())


def json_scalar(value: object) -> object:
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


def json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
