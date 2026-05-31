from __future__ import annotations

import hashlib
import json

import pandas as pd


DATA_AUDIT_SUMMARY_KEYS = [
    "data_audit_row_count",
    "data_audit_ok_count",
    "data_audit_failed_count",
    "data_audit_missing_file_count",
    "data_audit_missing_columns_count",
    "data_audit_no_window_data_count",
    "data_audit_quality_error_count",
    "data_audit_read_error_count",
    "data_min_coverage_threshold",
    "data_coverage_below_min_count",
    "data_coverage_below_min_ratio",
    "data_expected_rows",
    "data_missing_rows",
    "data_weighted_coverage_ratio",
    "data_min_coverage_ratio",
    "data_coverage_p05",
    "data_coverage_p50",
    "data_coverage_p95",
    "data_max_missing_gap_minutes",
    "data_max_missing_gap_start_at",
    "data_max_missing_gap_end_at",
    "data_zero_volume_amount_rows",
    "data_non_positive_price_rows",
    "data_negative_volume_amount_rows",
]

LIMIT_FILTER_SUMMARY_KEYS = [
    "limit_filter_audit_row_count",
    "limit_filter_enabled_count",
    "limit_filter_ok_count",
    "limit_filter_failed_count",
    "limit_filter_daily_missing_count",
    "limit_filter_daily_read_error_count",
    "limit_filter_daily_missing_columns_count",
    "limit_filter_daily_quality_error_count",
    "limit_filter_filtered_days",
]

DATA_INVENTORY_SUMMARY_KEYS = [
    "data_inventory_row_count",
    "data_inventory_cached_count",
    "data_inventory_unavailable_count",
    "data_inventory_missing_file_count",
    "data_inventory_read_error_count",
    "data_inventory_missing_columns_count",
    "data_inventory_no_valid_rows_count",
    "data_inventory_total_rows",
    "data_inventory_total_file_size_bytes",
    "data_inventory_signature",
]


def summarize_data_audit(audit: pd.DataFrame, *, min_coverage_ratio: float | None = None) -> dict[str, object]:
    """把行情审计表压成统计字段，便于 stats.json 和参数遍历表直接对比数据质量。"""
    if audit.empty:
        return _empty_data_audit_summary()
    min_coverage_ratio = normalize_min_coverage_ratio(min_coverage_ratio)
    status = (
        audit["status"].fillna("").astype(str)
        if "status" in audit.columns
        else pd.Series([""] * len(audit), index=audit.index)
    )
    expected_rows = _numeric_column(audit, "expected_rows")
    missing_rows = _numeric_column(audit, "missing_rows")
    coverage_ratio = _numeric_column(audit, "coverage_ratio")
    expected_mask = expected_rows.gt(0)
    below_min = _coverage_below_min_mask(expected_mask, coverage_ratio, min_coverage_ratio)
    expected_total = float(expected_rows.sum())
    missing_total = float(missing_rows.sum())
    max_missing_gap = _numeric_column(audit, "max_missing_gap_minutes")
    max_missing_gap_bounds = _max_missing_gap_bounds(audit, max_missing_gap)
    return {
        "data_audit_row_count": float(len(audit)),
        "data_audit_ok_count": float(status.eq("ok").sum()),
        "data_audit_failed_count": float(status.ne("ok").sum()),
        "data_audit_missing_file_count": float(status.eq("missing_file").sum()),
        "data_audit_missing_columns_count": float(status.eq("missing_columns").sum()),
        "data_audit_no_window_data_count": float(status.eq("no_window_data").sum()),
        "data_audit_quality_error_count": float(status.eq("quality_error").sum()),
        "data_audit_read_error_count": float(status.eq("read_error").sum()),
        "data_min_coverage_threshold": float(min_coverage_ratio or 0.0),
        "data_coverage_below_min_count": float(below_min.sum()),
        "data_coverage_below_min_ratio": _ratio_or_zero(float(below_min.sum()), float(expected_mask.sum())),
        "data_expected_rows": expected_total,
        "data_missing_rows": missing_total,
        "data_weighted_coverage_ratio": _ratio_or_zero(expected_total - missing_total, expected_total),
        "data_min_coverage_ratio": _min_or_zero(coverage_ratio.loc[expected_mask]),
        "data_coverage_p05": _quantile_or_zero(coverage_ratio.loc[expected_mask], 0.05),
        "data_coverage_p50": _quantile_or_zero(coverage_ratio.loc[expected_mask], 0.50),
        "data_coverage_p95": _quantile_or_zero(coverage_ratio.loc[expected_mask], 0.95),
        "data_max_missing_gap_minutes": _max_or_zero(max_missing_gap),
        **max_missing_gap_bounds,
        "data_zero_volume_amount_rows": float(_numeric_column(audit, "zero_volume_amount_rows").sum()),
        "data_non_positive_price_rows": float(_numeric_column(audit, "non_positive_price_rows").sum()),
        "data_negative_volume_amount_rows": float(_numeric_column(audit, "negative_volume_amount_rows").sum()),
    }


def summarize_limit_filter_audit(filter_audit: pd.DataFrame) -> dict[str, float]:
    """把日 K 涨停开盘过滤审计压成统计字段，避免只靠人工打开 CSV 判断。"""
    if filter_audit.empty:
        return {key: 0.0 for key in LIMIT_FILTER_SUMMARY_KEYS}
    status = (
        filter_audit["status"].fillna("").astype(str)
        if "status" in filter_audit.columns
        else pd.Series([""] * len(filter_audit), index=filter_audit.index)
    )
    enabled = (
        filter_audit["filter_enabled"].fillna(False).astype(bool)
        if "filter_enabled" in filter_audit.columns
        else pd.Series([False] * len(filter_audit), index=filter_audit.index)
    )
    return {
        "limit_filter_audit_row_count": float(len(filter_audit)),
        "limit_filter_enabled_count": float(enabled.sum()),
        "limit_filter_ok_count": float(status.eq("ok").sum()),
        "limit_filter_failed_count": float(status.ne("ok").sum()),
        "limit_filter_daily_missing_count": float(status.eq("daily_missing").sum()),
        "limit_filter_daily_read_error_count": float(status.eq("daily_read_error").sum()),
        "limit_filter_daily_missing_columns_count": float(status.eq("daily_missing_columns").sum()),
        "limit_filter_daily_quality_error_count": float(status.eq("daily_quality_error").sum()),
        "limit_filter_filtered_days": float(_numeric_column(filter_audit, "filtered_days").sum()),
    }


def summarize_data_inventory(data_inventory: pd.DataFrame | None) -> dict[str, object]:
    """把本地 parquet 库存压成稳定摘要；签名排除绝对路径，便于跨机器复现。"""
    keys: dict[str, object] = {key: 0.0 for key in DATA_INVENTORY_SUMMARY_KEYS}
    keys["data_inventory_signature"] = ""
    if data_inventory is None or data_inventory.empty:
        return keys
    status = (
        data_inventory["status"].fillna("").astype(str)
        if "status" in data_inventory.columns
        else pd.Series([""] * len(data_inventory), index=data_inventory.index)
    )
    rows = _numeric_column(data_inventory, "rows")
    file_size = _numeric_column(data_inventory, "file_size_bytes")
    cached = status.eq("cached")
    return {
        "data_inventory_row_count": float(len(data_inventory)),
        "data_inventory_cached_count": float(cached.sum()),
        "data_inventory_unavailable_count": float((~cached).sum()),
        "data_inventory_missing_file_count": float(status.eq("missing_file").sum()),
        "data_inventory_read_error_count": float(status.eq("read_error").sum()),
        "data_inventory_missing_columns_count": float(status.eq("missing_columns").sum()),
        "data_inventory_no_valid_rows_count": float(status.eq("no_valid_rows").sum()),
        "data_inventory_total_rows": float(rows.sum()),
        "data_inventory_total_file_size_bytes": float(file_size.sum()),
        "data_inventory_signature": _data_inventory_signature(data_inventory),
    }


def summarize_data_management(
    data_audit: pd.DataFrame,
    limit_filter_audit: pd.DataFrame,
    *,
    filtered_limit_open_count: int,
    data_inventory: pd.DataFrame | None = None,
    min_coverage_ratio: float | None = None,
) -> dict[str, object]:
    """汇总回测数据健康度；实验层、CLI 和 Web 都应复用这一组数据管理指标。"""
    stats = summarize_data_audit(data_audit, min_coverage_ratio=min_coverage_ratio)
    stats.update(summarize_data_inventory(data_inventory))
    stats.update(summarize_limit_filter_audit(limit_filter_audit))
    stats["filtered_limit_open_count"] = float(filtered_limit_open_count)
    return stats


def normalize_min_coverage_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if pd.isna(ratio) or ratio <= 0 or ratio > 1:
        raise ValueError("min_coverage_ratio 必须在 (0, 1] 之间。")
    return ratio


def _empty_data_audit_summary() -> dict[str, object]:
    summary = {key: 0.0 for key in DATA_AUDIT_SUMMARY_KEYS}
    summary["data_max_missing_gap_start_at"] = ""
    summary["data_max_missing_gap_end_at"] = ""
    return summary


def _max_missing_gap_bounds(audit: pd.DataFrame, max_missing_gap: pd.Series) -> dict[str, str]:
    if audit.empty or max_missing_gap.empty or max_missing_gap.max() <= 0:
        return {"data_max_missing_gap_start_at": "", "data_max_missing_gap_end_at": ""}
    row_index = max_missing_gap.idxmax()
    row = audit.loc[row_index]
    return {
        "data_max_missing_gap_start_at": _timestamp_label(row.get("max_missing_gap_start_at")),
        "data_max_missing_gap_end_at": _timestamp_label(row.get("max_missing_gap_end_at")),
    }


def _timestamp_label(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _data_inventory_signature(data_inventory: pd.DataFrame) -> str:
    """按缓存元数据生成跨机器稳定签名；不包含本机绝对路径。"""
    signature_columns = (
        "stock_code",
        "timeframe",
        "adjust",
        "status",
        "exists",
        "rows",
        "start",
        "end",
        "file_size_bytes",
        "missing_columns",
    )
    records = [
        {
            column: _inventory_signature_value(row.get(column))
            for column in signature_columns
        }
        for row in _sorted_inventory_records(data_inventory, signature_columns)
    ]
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sorted_inventory_records(data_inventory: pd.DataFrame, columns: tuple[str, ...]) -> list[dict[str, object]]:
    available = [column for column in columns if column in data_inventory.columns]
    if not available:
        return []
    frame = data_inventory.loc[:, available].copy()
    sort_columns = [column for column in ("stock_code", "timeframe", "adjust") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort")
    return frame.to_dict("records")


def _inventory_signature_value(value: object) -> object:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    return str(value)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(round(float(numerator) / float(denominator), 12))


def _coverage_below_min_mask(
    expected_mask: pd.Series,
    coverage_ratio: pd.Series,
    min_coverage_ratio: float | None,
) -> pd.Series:
    if min_coverage_ratio is None:
        return pd.Series([False] * len(coverage_ratio), index=coverage_ratio.index)
    return expected_mask & coverage_ratio.lt(min_coverage_ratio)


def _min_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.min()), 12))


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.max()), 12))


def _quantile_or_zero(values: pd.Series, quantile: float) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.quantile(quantile)), 12))
