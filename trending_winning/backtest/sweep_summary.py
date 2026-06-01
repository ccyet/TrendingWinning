from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import pandas as pd

from trending_winning.backtest.sweep_analysis import json_scalar, truthy_column_count
from trending_winning.data.summary import DATA_INVENTORY_SUMMARY_KEYS


SWEEP_SUMMARY_CONTEXT_COLUMNS = (
    *DATA_INVENTORY_SUMMARY_KEYS,
    "primary_data_issue",
    "primary_data_issue_count",
    "primary_data_issue_rate",
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


def sweep_summary_statistics(
    *,
    table: pd.DataFrame,
    grid: Mapping[str, Sequence[object]],
    elapsed_seconds: float,
    input_bar_count: int,
    filtered_limit_open_count: int,
    strategy_stats: pd.DataFrame,
    detector_stats: pd.DataFrame,
    setup_stats: pd.DataFrame,
    symbol_stats: pd.DataFrame,
    setup_order_decision_stats: pd.DataFrame,
    setup_strategy_filter_stats: pd.DataFrame,
    case_diagnostics: pd.DataFrame | None = None,
) -> dict[str, object]:
    """把参数遍历压成一份总览 JSON，便于 Web/CLI 快速展示。"""
    summary: dict[str, object] = {
        "case_count": int(len(table)),
        "grid_case_count": int(math.prod(grid_value_counts(grid).values())) if grid else 0,
        "grid_field_count": int(len(grid)),
        "grid_fields": list(grid),
        "grid_value_counts": grid_value_counts(grid),
        "pareto_case_count": truthy_column_count(table, "is_pareto_efficient"),
        "elapsed_seconds": float(elapsed_seconds),
        "input_bar_count": int(input_bar_count),
        "filtered_limit_open_count": int(filtered_limit_open_count),
        "best_case_name": "",
        "best_case_config_hash": "",
        **risk_adjusted_summary_statistics(table),
    }
    if not table.empty:
        best = table.iloc[0]
        summary["best_case_name"] = str(best.get("case_name", ""))
        summary["best_case_config_hash"] = str(best.get("case_config_hash", ""))
        for column in SWEEP_SUMMARY_BEST_COLUMNS:
            if column in table.columns:
                summary[f"best_{column}"] = json_scalar(best[column])
        for column in SWEEP_SUMMARY_CONTEXT_COLUMNS:
            if column in table.columns:
                summary[column] = json_scalar(best[column])

    summary.update(cache_status_statistics(table, "order_cache_status", prefix="order_cache"))
    summary.update(cache_status_statistics(table, "candidate_cache_status", prefix="candidate_cache"))
    summary.update(case_trade_summary_statistics(strategy_stats, prefix="case_strategy"))
    summary.update(case_trade_summary_statistics(detector_stats, prefix="case_detector"))
    summary.update(case_trade_summary_statistics(setup_stats, prefix="case_setup"))
    summary.update(case_trade_summary_statistics(symbol_stats, prefix="case_symbol"))
    summary.update(case_decision_summary_statistics(setup_order_decision_stats, prefix="case_setup_order"))
    summary.update(case_decision_summary_statistics(setup_strategy_filter_stats, prefix="case_setup_strategy_filter"))
    summary.update(case_diagnostic_summary_statistics(case_diagnostics))
    for column in ("generated_order_count", "candidate_count", "candidate_rejection_count"):
        if column in table.columns:
            summary[column] = numeric_column_sum(table, column)
    return summary


def grid_value_counts(grid: Mapping[str, Sequence[object]]) -> dict[str, int]:
    return {str(key): len(list(values)) for key, values in grid.items()}


def cache_status_statistics(table: pd.DataFrame, column: str, *, prefix: str) -> dict[str, float]:
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


def case_trade_summary_statistics(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
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


def case_decision_summary_statistics(frame: pd.DataFrame, *, prefix: str) -> dict[str, float]:
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


def case_diagnostic_summary_statistics(frame: pd.DataFrame | None) -> dict[str, float]:
    keys = {
        "case_diagnostic_row_count": 0.0,
        "case_diagnostic_failed_count": 0.0,
        "case_diagnostic_attention_count": 0.0,
        "case_diagnostic_passed_count": 0.0,
        "case_diagnostic_failed_case_count": 0.0,
        "case_diagnostic_attention_case_count": 0.0,
    }
    if frame is None or frame.empty or "status" not in frame.columns:
        return keys
    status = frame["status"].fillna("").astype(str)
    keys["case_diagnostic_row_count"] = float(len(frame))
    keys["case_diagnostic_failed_count"] = float(status.eq("失败").sum())
    keys["case_diagnostic_attention_count"] = float(status.eq("关注").sum())
    keys["case_diagnostic_passed_count"] = float(status.eq("通过").sum())
    if "case_name" in frame.columns:
        keys["case_diagnostic_failed_case_count"] = float(frame.loc[status.eq("失败"), "case_name"].astype(str).nunique())
        keys["case_diagnostic_attention_case_count"] = float(frame.loc[status.eq("关注"), "case_name"].astype(str).nunique())
    return keys


def risk_adjusted_summary_statistics(table: pd.DataFrame) -> dict[str, object]:
    """汇总风险质量评分，便于 summary.json 直接定位稳健参数组。"""
    result: dict[str, object] = {
        "best_risk_adjusted_case_name": "",
        "best_risk_adjusted_case_config_hash": "",
        "best_risk_adjusted_sweep_rank": 0,
        "best_risk_adjusted_score": 0.0,
        "avg_risk_adjusted_score": 0.0,
        "median_risk_adjusted_score": 0.0,
        "worst_risk_adjusted_score": 0.0,
    }
    if table.empty or "risk_adjusted_score" not in table.columns:
        return result

    score = pd.to_numeric(table["risk_adjusted_score"], errors="coerce")
    valid = score.dropna()
    if valid.empty:
        return result

    best_index = valid.idxmax()
    best = table.loc[best_index]
    result.update(
        {
            "best_risk_adjusted_case_name": str(best.get("case_name", "")),
            "best_risk_adjusted_case_config_hash": str(best.get("case_config_hash", "")),
            "best_risk_adjusted_sweep_rank": json_scalar(best.get("sweep_rank", 0)),
            "best_risk_adjusted_score": json_scalar(score.loc[best_index]),
            "avg_risk_adjusted_score": float(valid.mean()),
            "median_risk_adjusted_score": float(valid.median()),
            "worst_risk_adjusted_score": float(valid.min()),
        }
    )
    return result


def numeric_column_sum(table: pd.DataFrame, column: str) -> float:
    if table.empty or column not in table.columns:
        return 0.0
    return float(pd.to_numeric(table[column], errors="coerce").fillna(0.0).sum())
