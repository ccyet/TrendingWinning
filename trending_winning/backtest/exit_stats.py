from __future__ import annotations

import pandas as pd


EXIT_REASONS = (
    "take_profit",
    "trailing_take_profit",
    "stop_loss",
    "max_holding",
    "end_of_data",
)

EXIT_REASON_STAT_KEYS = tuple(
    key
    for reason in (*EXIT_REASONS, "other")
    for key in (f"{reason}_exit_count", f"{reason}_exit_rate")
)


def summarize_exit_reasons(trades: pd.DataFrame) -> dict[str, float]:
    """汇总平仓原因分布；固定字段便于 stats.json 和参数遍历直接对比。"""
    if trades.empty or "exit_reason" not in trades.columns:
        return {key: 0.0 for key in EXIT_REASON_STAT_KEYS}

    reason = trades["exit_reason"].fillna("").astype(str)
    total = float(len(reason))
    known = set(EXIT_REASONS)
    result: dict[str, float] = {}
    for exit_reason in EXIT_REASONS:
        count = float(reason.eq(exit_reason).sum())
        result[f"{exit_reason}_exit_count"] = count
        result[f"{exit_reason}_exit_rate"] = _ratio_or_zero(count, total)

    other_count = float((~reason.isin(known)).sum())
    result["other_exit_count"] = other_count
    result["other_exit_rate"] = _ratio_or_zero(other_count, total)
    return result


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
