from __future__ import annotations

import pandas as pd


def trade_path_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """按真实平仓时间排序交易路径；没有时间字段时保留调用方给定顺序。"""
    realized_at = coalesced_datetime_columns(trades, ("exit_date", "entry_date", "signal_date"))
    if realized_at is None or realized_at.isna().all():
        return trades

    result = trades.copy()
    result["_realized_at"] = realized_at.to_numpy()
    result["_row_order"] = range(len(result))
    sort_columns = ["_realized_at", "_row_order"]
    if "trade_no" in result.columns:
        result["_trade_no_sort"] = pd.to_numeric(result["trade_no"], errors="coerce")
        sort_columns = ["_realized_at", "_trade_no_sort", "_row_order"]
    return (
        result.sort_values(sort_columns, kind="mergesort", na_position="last")
        .drop(columns=[column for column in ("_realized_at", "_trade_no_sort", "_row_order") if column in result.columns])
        .reset_index(drop=True)
    )


def coalesced_datetime_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series | None:
    """按优先级合并多个日期列，避免单列缺失导致时间轴丢失。"""
    result: pd.Series | None = None
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").reset_index(drop=True)
        result = values if result is None else result.fillna(values)
    return result


def trade_returns_as_decimal(trades: pd.DataFrame) -> pd.Series:
    """把逐笔百分比收益转成小数收益，供净值曲线和统计指标共用。"""
    return pd.to_numeric(trades.get("return_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) / 100.0


def equity_with_initial_point(returns: pd.Series) -> pd.Series:
    """把逐笔小数收益转成从 1.0 起步的净值路径。"""
    equity = (1.0 + returns).cumprod()
    return pd.concat([pd.Series([1.0]), equity], ignore_index=True)
