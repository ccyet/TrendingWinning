from __future__ import annotations

import re

import pandas as pd


DECISION_METRIC_COLUMNS = {
    "actual_risk_pct": ("avg_actual_risk_pct", "max_actual_risk_pct"),
    "actual_chase_pct": ("avg_actual_chase_pct", "max_actual_chase_pct"),
    "actual_reward_to_risk": ("avg_actual_reward_to_risk", "min_actual_reward_to_risk"),
}


def compute_decision_reason_statistics(
    decisions: pd.DataFrame,
    *,
    group_fields: tuple[str, ...] = ("strategy_name", "detector_name"),
) -> pd.DataFrame:
    """按策略、状态和原因汇总决策分布；输入只需要决策表，不依赖策略实现。"""
    metric_columns = _decision_metric_output_columns()
    columns = pd.Index(
        [
            *group_fields,
            "status",
            "reason",
            "decision_count",
            "decision_rate",
            "group_decision_count",
            "group_decision_rate",
            *metric_columns,
        ]
    )
    if decisions.empty:
        return pd.DataFrame(columns=columns)
    required = {*group_fields, "status"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decisions 缺少分组字段：{', '.join(sorted(missing))}")

    data = decisions.copy()
    for column in group_fields:
        data[column] = data[column].fillna("").astype(str)
    data["status"] = data["status"].fillna("").astype(str)
    data["reason"] = data["reason"].fillna("").astype(str) if "reason" in data.columns else ""
    for source_column in DECISION_METRIC_COLUMNS:
        data[source_column] = _decision_metric_series(data, source_column)

    total = len(data)
    group_keys = [*group_fields, "status", "reason"]
    grouped = data.groupby(group_keys, sort=True, dropna=False).size().reset_index(name="decision_count")
    grouped["decision_count"] = grouped["decision_count"].astype(int)
    grouped["decision_rate"] = grouped["decision_count"].map(lambda value: _ratio_or_zero(float(value), float(total)))
    grouped = _attach_group_decision_rates(data, grouped, group_fields=group_fields)
    metric_stats = _decision_metric_statistics(data, group_keys)
    grouped = grouped.merge(metric_stats, on=group_keys, how="left")
    grouped[metric_columns] = grouped[metric_columns].fillna(0.0)
    return grouped[columns]


def summarize_order_decisions(order_decisions: pd.DataFrame) -> dict[str, object]:
    """汇总订单接受和拒绝原因；用于解释信号为什么没有变成成交。"""
    keys = [
        "order_count",
        "accepted_order_count",
        "rejected_order_count",
        "acceptance_rate",
        "rejection_rate",
        "primary_rejected_reason",
        "primary_rejected_reason_count",
        "primary_rejected_reason_rate",
        "rejected_no_fill_count",
        "rejected_no_liquidity_count",
        "rejected_no_bars_count",
        "rejected_invalid_order_count",
        "rejected_duplicate_order_id_count",
        "rejected_already_open_count",
        "rejected_max_open_positions_count",
        "rejected_same_symbol_overlap_count",
        "rejected_no_capital_count",
        "rejected_actual_risk_too_high_count",
        "rejected_chase_too_far_count",
        "rejected_target_not_favorable_count",
        "avg_accepted_capital_fraction",
        "max_accepted_capital_fraction",
        "avg_accepted_risk_fraction",
        "max_accepted_risk_fraction",
        "avg_accepted_margin_fraction",
        "max_accepted_margin_fraction",
        "executed_order_count",
        "accepted_executed_order_count",
        "avg_accepted_actual_risk_pct",
        "max_accepted_actual_risk_pct",
        "avg_accepted_actual_chase_pct",
        "max_accepted_actual_chase_pct",
        "avg_accepted_actual_reward_to_risk",
        "min_accepted_actual_reward_to_risk",
        "avg_executed_actual_risk_pct",
        "max_executed_actual_risk_pct",
        "avg_executed_actual_chase_pct",
        "max_executed_actual_chase_pct",
        "avg_executed_actual_reward_to_risk",
        "min_executed_actual_reward_to_risk",
    ]
    if order_decisions.empty or "status" not in order_decisions.columns:
        return _empty_summary(keys, text_keys={"primary_rejected_reason"})

    status = order_decisions["status"].astype(str)
    reason = order_decisions["reason"].astype(str) if "reason" in order_decisions.columns else _empty_reason_series(order_decisions)
    order_count = float(len(order_decisions))
    accepted = status.eq("accepted")
    rejected = status.eq("rejected")
    accepted_count = float(accepted.sum())
    rejected_count = float(rejected.sum())
    executed = _executed_decisions(order_decisions)
    accepted_executed = accepted & executed
    result = {
        "order_count": order_count,
        "accepted_order_count": accepted_count,
        "rejected_order_count": rejected_count,
        "acceptance_rate": _ratio_or_zero(accepted_count, order_count),
        "rejection_rate": _ratio_or_zero(rejected_count, order_count),
        **_primary_rejected_reason_fields(
            reason,
            rejected,
            reason_key="primary_rejected_reason",
            count_key="primary_rejected_reason_count",
            rate_key="primary_rejected_reason_rate",
        ),
        "rejected_no_fill_count": float((rejected & reason.eq("no_fill")).sum()),
        "rejected_no_liquidity_count": float((rejected & reason.eq("no_liquidity")).sum()),
        "rejected_no_bars_count": float((rejected & reason.eq("no_bars")).sum()),
        "rejected_invalid_order_count": float((rejected & reason.eq("invalid_order")).sum()),
        "rejected_duplicate_order_id_count": float((rejected & reason.eq("duplicate_order_id")).sum()),
        "rejected_already_open_count": float((rejected & reason.eq("already_open")).sum()),
        "rejected_max_open_positions_count": float((rejected & reason.eq("max_open_positions")).sum()),
        "rejected_same_symbol_overlap_count": float((rejected & reason.eq("same_symbol_overlap")).sum()),
        "rejected_no_capital_count": float((rejected & reason.eq("no_capital")).sum()),
        "rejected_actual_risk_too_high_count": float((rejected & reason.eq("actual_risk_too_high")).sum()),
        "rejected_chase_too_far_count": float((rejected & reason.eq("chase_too_far")).sum()),
        "rejected_target_not_favorable_count": float((rejected & reason.eq("target_not_favorable")).sum()),
        "avg_accepted_capital_fraction": _accepted_mean(order_decisions, accepted, "capital_fraction"),
        "max_accepted_capital_fraction": _accepted_max(order_decisions, accepted, "capital_fraction"),
        "avg_accepted_risk_fraction": _accepted_mean(order_decisions, accepted, "risk_fraction"),
        "max_accepted_risk_fraction": _accepted_max(order_decisions, accepted, "risk_fraction"),
        "avg_accepted_margin_fraction": _accepted_mean(order_decisions, accepted, "margin_fraction"),
        "max_accepted_margin_fraction": _accepted_max(order_decisions, accepted, "margin_fraction"),
        "executed_order_count": float(executed.sum()),
        "accepted_executed_order_count": float(accepted_executed.sum()),
        "avg_accepted_actual_risk_pct": _masked_mean(order_decisions, accepted_executed, "actual_risk_pct"),
        "max_accepted_actual_risk_pct": _masked_max(order_decisions, accepted_executed, "actual_risk_pct"),
        "avg_accepted_actual_chase_pct": _masked_mean(order_decisions, accepted_executed, "actual_chase_pct"),
        "max_accepted_actual_chase_pct": _masked_max(order_decisions, accepted_executed, "actual_chase_pct"),
        "avg_accepted_actual_reward_to_risk": _masked_mean(order_decisions, accepted_executed, "actual_reward_to_risk"),
        "min_accepted_actual_reward_to_risk": _masked_min(order_decisions, accepted_executed, "actual_reward_to_risk"),
        "avg_executed_actual_risk_pct": _masked_mean(order_decisions, executed, "actual_risk_pct"),
        "max_executed_actual_risk_pct": _masked_max(order_decisions, executed, "actual_risk_pct"),
        "avg_executed_actual_chase_pct": _masked_mean(order_decisions, executed, "actual_chase_pct"),
        "max_executed_actual_chase_pct": _masked_max(order_decisions, executed, "actual_chase_pct"),
        "avg_executed_actual_reward_to_risk": _masked_mean(order_decisions, executed, "actual_reward_to_risk"),
        "min_executed_actual_reward_to_risk": _masked_min(order_decisions, executed, "actual_reward_to_risk"),
    }
    result.update(_rejected_reason_counts(reason, rejected, prefix="rejected"))
    return result


def summarize_strategy_filter_decisions(filter_decisions: pd.DataFrame) -> dict[str, object]:
    """汇总策略层过滤结果；用于解释信号为什么没有进入撮合层。"""
    keys = [
        "strategy_signal_count",
        "strategy_accepted_signal_count",
        "strategy_rejected_signal_count",
        "strategy_filter_acceptance_rate",
        "strategy_filter_rejection_rate",
        "primary_strategy_rejected_reason",
        "primary_strategy_rejected_reason_count",
        "primary_strategy_rejected_reason_rate",
        "strategy_rejected_higher_timeframe_mismatch_count",
        "strategy_rejected_higher_timeframe_no_context_count",
        "strategy_rejected_higher_timeframe_stale_count",
        "strategy_rejected_invalid_order_key_count",
        "strategy_rejected_signal_bar_no_liquidity_count",
    ]
    if filter_decisions.empty or "status" not in filter_decisions.columns:
        return _empty_summary(keys, text_keys={"primary_strategy_rejected_reason"})

    status = filter_decisions["status"].astype(str)
    reason = filter_decisions["reason"].astype(str) if "reason" in filter_decisions.columns else _empty_reason_series(filter_decisions)
    signal_count = float(len(filter_decisions))
    accepted = status.eq("accepted")
    rejected = status.eq("rejected")
    accepted_count = float(accepted.sum())
    rejected_count = float(rejected.sum())
    result = {
        "strategy_signal_count": signal_count,
        "strategy_accepted_signal_count": accepted_count,
        "strategy_rejected_signal_count": rejected_count,
        "strategy_filter_acceptance_rate": _ratio_or_zero(accepted_count, signal_count),
        "strategy_filter_rejection_rate": _ratio_or_zero(rejected_count, signal_count),
        **_primary_rejected_reason_fields(
            reason,
            rejected,
            reason_key="primary_strategy_rejected_reason",
            count_key="primary_strategy_rejected_reason_count",
            rate_key="primary_strategy_rejected_reason_rate",
        ),
        "strategy_rejected_higher_timeframe_mismatch_count": float((rejected & reason.eq("higher_timeframe_mismatch")).sum()),
        "strategy_rejected_higher_timeframe_no_context_count": float((rejected & reason.eq("higher_timeframe_no_context")).sum()),
        "strategy_rejected_higher_timeframe_stale_count": float((rejected & reason.eq("higher_timeframe_stale")).sum()),
        "strategy_rejected_invalid_order_key_count": float((rejected & reason.eq("invalid_order_key")).sum()),
        "strategy_rejected_signal_bar_no_liquidity_count": float((rejected & reason.eq("signal_bar_no_liquidity")).sum()),
    }
    result.update(_rejected_reason_counts(reason, rejected, prefix="strategy_rejected"))
    return result


def _attach_group_decision_rates(
    data: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    group_fields: tuple[str, ...],
) -> pd.DataFrame:
    if not group_fields:
        result = grouped.copy()
        result["group_decision_count"] = int(len(data))
    else:
        group_totals = data.groupby(list(group_fields), sort=True, dropna=False).size().reset_index(name="group_decision_count")
        result = grouped.merge(group_totals, on=list(group_fields), how="left")
    result["group_decision_count"] = result["group_decision_count"].fillna(0).astype(int)
    result["group_decision_rate"] = [
        _ratio_or_zero(float(decision_count), float(group_count))
        for decision_count, group_count in zip(result["decision_count"], result["group_decision_count"], strict=False)
    ]
    return result


def _decision_metric_output_columns() -> list[str]:
    columns: list[str] = []
    for output_columns in DECISION_METRIC_COLUMNS.values():
        columns.extend(output_columns)
    return columns


def _decision_metric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series([0.0] * len(data), index=data.index, dtype=float)
    return pd.to_numeric(data[column], errors="coerce").fillna(0.0)


def _decision_metric_statistics(data: pd.DataFrame, group_keys: list[str]) -> pd.DataFrame:
    grouped = data.groupby(group_keys, sort=True, dropna=False)
    stats = grouped.agg(
        avg_actual_risk_pct=("actual_risk_pct", "mean"),
        max_actual_risk_pct=("actual_risk_pct", "max"),
        avg_actual_chase_pct=("actual_chase_pct", "mean"),
        max_actual_chase_pct=("actual_chase_pct", "max"),
        avg_actual_reward_to_risk=("actual_reward_to_risk", "mean"),
        min_actual_reward_to_risk=("actual_reward_to_risk", "min"),
    ).reset_index()
    for column in _decision_metric_output_columns():
        stats[column] = stats[column].map(_round_float)
    return stats


def _rejected_reason_counts(reason: pd.Series, rejected: pd.Series, *, prefix: str) -> dict[str, float]:
    rejected_reasons = reason.loc[rejected].fillna("").astype(str)
    rejected_reasons = rejected_reasons.loc[rejected_reasons.ne("")]
    if rejected_reasons.empty:
        return {}
    counts = rejected_reasons.value_counts(sort=True)
    return {
        f"{prefix}_{_metric_safe_reason(str(reason_value))}_count": float(count)
        for reason_value, count in counts.items()
        if _metric_safe_reason(str(reason_value))
    }


def _primary_rejected_reason_fields(
    reason: pd.Series,
    rejected: pd.Series,
    *,
    reason_key: str,
    count_key: str,
    rate_key: str,
) -> dict[str, object]:
    rejected_reasons = reason.loc[rejected].fillna("").astype(str)
    rejected_reasons = rejected_reasons.loc[rejected_reasons.ne("")]
    if rejected_reasons.empty:
        return {reason_key: "", count_key: 0.0, rate_key: 0.0}
    counts = rejected_reasons.value_counts(sort=True)
    primary_reason = str(counts.index[0])
    primary_count = float(counts.iloc[0])
    return {
        reason_key: primary_reason,
        count_key: primary_count,
        rate_key: _ratio_or_zero(primary_count, float(len(rejected_reasons))),
    }


def _empty_summary(keys: list[str], *, text_keys: set[str] | None = None) -> dict[str, object]:
    text_keys = text_keys or set()
    return {key: "" if key in text_keys else 0.0 for key in keys}


def _metric_safe_reason(reason: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", reason.strip().lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def _empty_reason_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([""] * len(frame), index=frame.index, dtype=str)


def _executed_decisions(frame: pd.DataFrame) -> pd.Series:
    if "actual_entry_price" not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    entry_price = pd.to_numeric(frame["actual_entry_price"], errors="coerce").fillna(0.0)
    return entry_price > 0


def _accepted_values(frame: pd.DataFrame, accepted: pd.Series, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame.loc[accepted, column], errors="coerce").dropna().astype(float).reset_index(drop=True)


def _accepted_mean(frame: pd.DataFrame, accepted: pd.Series, column: str) -> float:
    return _mean_or_zero(_accepted_values(frame, accepted, column))


def _accepted_max(frame: pd.DataFrame, accepted: pd.Series, column: str) -> float:
    values = _accepted_values(frame, accepted, column)
    return _round_float(values.max()) if not values.empty else 0.0


def _masked_values(frame: pd.DataFrame, mask: pd.Series, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame.loc[mask, column], errors="coerce").dropna().astype(float).reset_index(drop=True)


def _masked_mean(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    return _mean_or_zero(_masked_values(frame, mask, column))


def _masked_max(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = _masked_values(frame, mask, column)
    return _round_float(values.max()) if not values.empty else 0.0


def _masked_min(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    values = _masked_values(frame, mask, column)
    return _round_float(values.min()) if not values.empty else 0.0


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
