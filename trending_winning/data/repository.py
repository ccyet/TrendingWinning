from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trending_winning.data.filters import filter_limit_open_days, limit_open_dates
from trending_winning.data.schema import (
    CANONICAL_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_DIR_NAMES,
    empty_bars,
    empty_download_result,
    ensure_supported_timeframe,
    inclusive_end_timestamp,
    normalize_bars,
    normalize_symbol,
    parse_time_window,
    resolve_timeframe_root,
    unique_symbols,
)

__all__ = [
    "BacktestDataBundle",
    "MultiTimeframeBacktestDataBundle",
    "MarketDataRepository",
    "audit_local_data",
    "available_symbols",
    "inventory_local_data",
    "load_backtest_data",
    "load_daily_bars",
    "load_local_bars",
    "load_multi_timeframe_backtest_data",
    "plan_tdx_backtest_data",
    "prepare_tdx_backtest_data",
    "resolve_daily_root",
    "resolve_timeframe_root",
    "summarize_data_audit",
    "summarize_limit_filter_audit",
    "update_from_tdx",
    "write_local_bars",
]

AUDIT_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "status",
    "exists",
    "rows_total",
    "rows_in_window",
    "expected_rows",
    "missing_rows",
    "coverage_ratio",
    "max_missing_gap_minutes",
    "first_missing_at",
    "last_missing_at",
    "max_missing_gap_start_at",
    "max_missing_gap_end_at",
    "start",
    "end",
    "requested_start",
    "requested_end",
    "invalid_date_rows",
    "invalid_symbol_rows",
    "duplicate_rows",
    "null_ohlc_rows",
    "non_positive_price_rows",
    "inconsistent_ohlc_rows",
    "null_volume_amount_rows",
    "zero_volume_amount_rows",
    "negative_volume_amount_rows",
    "missing_columns",
    "path",
    "message",
]

PREPARE_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "action",
    "before_status",
    "after_status",
    "rows_written",
    "new_rows",
    "before_coverage_ratio",
    "after_coverage_ratio",
    "coverage_ratio",
    "before_missing_rows",
    "after_missing_rows",
    "missing_rows",
    "before_max_missing_gap_minutes",
    "after_max_missing_gap_minutes",
    "before_first_missing_at",
    "before_last_missing_at",
    "after_first_missing_at",
    "after_last_missing_at",
    "first_missing_at",
    "last_missing_at",
    "before_max_missing_gap_start_at",
    "before_max_missing_gap_end_at",
    "after_max_missing_gap_start_at",
    "after_max_missing_gap_end_at",
    "max_missing_gap_start_at",
    "max_missing_gap_end_at",
    "path",
    "message",
]

PLAN_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "action",
    "reason",
    "before_status",
    "rows_in_window",
    "expected_rows",
    "missing_rows",
    "coverage_ratio",
    "max_missing_gap_minutes",
    "first_missing_at",
    "last_missing_at",
    "max_missing_gap_start_at",
    "max_missing_gap_end_at",
    "path",
    "message",
]

LIMIT_FILTER_AUDIT_COLUMNS = [
    "stock_code",
    "status",
    "filter_enabled",
    "daily_rows",
    "filtered_days",
    "message",
]

INVENTORY_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "status",
    "exists",
    "rows",
    "start",
    "end",
    "file_size_bytes",
    "modified_at",
    "path",
    "message",
]

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
    "limit_filter_filtered_days",
]


@dataclass(frozen=True)
class BacktestDataBundle:
    """回测数据包；分钟线、日线、过滤日和数据审计结果一起返回。"""

    bars: pd.DataFrame
    daily_bars: pd.DataFrame
    filtered_limit_open_days: pd.DataFrame
    data_audit: pd.DataFrame
    limit_filter_audit: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=LIMIT_FILTER_AUDIT_COLUMNS)
    )


@dataclass(frozen=True)
class MultiTimeframeBacktestDataBundle:
    """多周期回测数据包；一次请求返回每个周期独立 K 线和统一日线过滤信息。"""

    bars_by_timeframe: dict[str, pd.DataFrame]
    daily_bars: pd.DataFrame
    filtered_limit_open_days: pd.DataFrame
    data_audit: pd.DataFrame
    limit_filter_audit: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=LIMIT_FILTER_AUDIT_COLUMNS)
    )


class MarketDataRepository:
    """本地行情仓库入口；统一读取分钟线、日线和写入 parquet。"""

    def __init__(self, data_root: str | Path, adjust: str = "qfq") -> None:
        self.data_root = Path(data_root).expanduser()
        self.adjust = adjust

    def available_symbols(self, timeframe: str) -> list[str]:
        return available_symbols(self.data_root, timeframe, self.adjust)

    def inventory(
        self,
        *,
        timeframes: tuple[str, ...] | list[str] = SUPPORTED_TIMEFRAMES,
        symbols: tuple[str, ...] | list[str] | None = None,
    ) -> pd.DataFrame:
        return inventory_local_data(
            data_root=self.data_root,
            adjust=self.adjust,
            timeframes=timeframes,
            symbols=symbols,
        )

    def load_bars(
        self,
        *,
        timeframe: str,
        symbols: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        return load_local_bars(
            data_root=self.data_root,
            timeframe=timeframe,
            adjust=self.adjust,
            symbols=symbols,
            start=start,
            end=end,
        )

    def load_daily_bars(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        return load_daily_bars(
            data_root=self.data_root,
            adjust=self.adjust,
            symbols=symbols,
            start=start,
            end=end,
        )

    def write_bars(self, *, timeframe: str, bars: pd.DataFrame) -> pd.DataFrame:
        return write_local_bars(data_root=self.data_root, timeframe=timeframe, adjust=self.adjust, bars=bars)

    def audit_bars(
        self,
        *,
        timeframe: str,
        symbols: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> pd.DataFrame:
        normalized_timeframe = ensure_supported_timeframe(timeframe)
        expected_sessions_by_symbol = (
            _expected_sessions_by_symbol_from_daily(
                data_root=self.data_root,
                adjust=self.adjust,
                symbols=unique_symbols(tuple(symbols)),
                start=start,
                end=end,
            )
            if normalized_timeframe != "1d"
            else None
        )
        return audit_local_data(
            data_root=self.data_root,
            timeframe=normalized_timeframe,
            adjust=self.adjust,
            symbols=symbols,
            start=start,
            end=end,
            expected_sessions_by_symbol=expected_sessions_by_symbol,
        )

    def update_from_tdx(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        timeframe: str,
        start: str,
        end: str,
        tqcenter_path: str = "",
        tq_client: Any | None = None,
    ) -> pd.DataFrame:
        return update_from_tdx(
            data_root=self.data_root,
            adjust=self.adjust,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            tqcenter_path=tqcenter_path,
            tq_client=tq_client,
        )

    def prepare_from_tdx(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        timeframes: tuple[str, ...] | list[str],
        start: str,
        end: str,
        tqcenter_path: str = "",
        tq_client: Any | None = None,
        min_coverage_ratio: float | None = None,
        strict_after_update: bool = True,
    ) -> pd.DataFrame:
        return prepare_tdx_backtest_data(
            data_root=self.data_root,
            adjust=self.adjust,
            symbols=symbols,
            timeframes=timeframes,
            start=start,
            end=end,
            tqcenter_path=tqcenter_path,
            tq_client=tq_client,
            min_coverage_ratio=min_coverage_ratio,
            strict_after_update=strict_after_update,
        )

    def plan_from_tdx(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        timeframes: tuple[str, ...] | list[str],
        start: str,
        end: str,
        min_coverage_ratio: float | None = None,
    ) -> pd.DataFrame:
        return plan_tdx_backtest_data(
            data_root=self.data_root,
            adjust=self.adjust,
            symbols=symbols,
            timeframes=timeframes,
            start=start,
            end=end,
            min_coverage_ratio=min_coverage_ratio,
        )

    def load_backtest_data(
        self,
        *,
        timeframe: str,
        symbols: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        filter_limit_open: bool = True,
        daily_lookback_days: int = 10,
        strict_data_quality: bool = True,
        min_coverage_ratio: float | None = None,
    ) -> BacktestDataBundle:
        return load_backtest_data(
            data_root=self.data_root,
            timeframe=timeframe,
            adjust=self.adjust,
            symbols=symbols,
            start=start,
            end=end,
            filter_limit_open=filter_limit_open,
            daily_lookback_days=daily_lookback_days,
            strict_data_quality=strict_data_quality,
            min_coverage_ratio=min_coverage_ratio,
        )

    def load_multi_timeframe_backtest_data(
        self,
        *,
        timeframes: tuple[str, ...] | list[str],
        symbols: tuple[str, ...] | list[str],
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
        filter_limit_open: bool = True,
        daily_lookback_days: int = 10,
        strict_data_quality: bool = True,
        min_coverage_ratio: float | None = None,
    ) -> MultiTimeframeBacktestDataBundle:
        return load_multi_timeframe_backtest_data(
            data_root=self.data_root,
            timeframes=timeframes,
            adjust=self.adjust,
            symbols=symbols,
            start=start,
            end=end,
            filter_limit_open=filter_limit_open,
            daily_lookback_days=daily_lookback_days,
            strict_data_quality=strict_data_quality,
            min_coverage_ratio=min_coverage_ratio,
        )


def available_symbols(data_root: str | Path, timeframe: str, adjust: str = "qfq") -> list[str]:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    if not root.exists():
        return []
    return sorted(normalize_symbol(path.stem) for path in root.glob("*.parquet"))


def inventory_local_data(
    *,
    data_root: str | Path,
    adjust: str = "qfq",
    timeframes: tuple[str, ...] | list[str] = SUPPORTED_TIMEFRAMES,
    symbols: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """列出本地 parquet 缓存库存；用于回测前确认哪些周期和标的已经落地。"""
    normalized_timeframes = _unique_timeframes(list(timeframes))
    if not normalized_timeframes:
        raise ValueError("timeframes 不能为空。")
    normalized_symbols = (
        unique_symbols(tuple(symbols))
        if symbols is not None
        else _discover_inventory_symbols(data_root=data_root, adjust=adjust, timeframes=normalized_timeframes)
    )
    if not normalized_symbols:
        return pd.DataFrame(columns=INVENTORY_COLUMNS)

    rows = [
        _inventory_symbol_file(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbol=symbol,
        )
        for timeframe in normalized_timeframes
        for symbol in normalized_symbols
    ]
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    order = {timeframe: index for index, timeframe in enumerate(SUPPORTED_TIMEFRAMES)}
    frame["_timeframe_order"] = frame["timeframe"].map(order).fillna(len(order)).astype(int)
    return (
        frame.sort_values(["_timeframe_order", "stock_code"], kind="mergesort")
        .drop(columns=["_timeframe_order"])
        .reset_index(drop=True)
    )


def _discover_inventory_symbols(
    *,
    data_root: str | Path,
    adjust: str,
    timeframes: list[str],
) -> list[str]:
    """未指定代码时按已存在的 parquet 文件反推标的清单。"""
    seen: set[str] = set()
    symbols: list[str] = []
    for timeframe in timeframes:
        root = resolve_timeframe_root(data_root, timeframe) / adjust
        if not root.exists():
            continue
        for path in sorted(root.glob("*.parquet")):
            symbol = normalize_symbol(path.stem)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return sorted(symbols)


def _inventory_symbol_file(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbol: str,
) -> dict[str, object]:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    path = root / f"{symbol}.parquet"
    base = {
        "stock_code": symbol,
        "timeframe": ensure_supported_timeframe(timeframe),
        "adjust": adjust,
        "exists": path.exists(),
        "file_size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "modified_at": pd.Timestamp.fromtimestamp(path.stat().st_mtime) if path.exists() else pd.NaT,
        "path": str(path),
    }
    if not path.exists():
        return _inventory_record(base, status="missing_file", message="本地 parquet 不存在。")
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return _inventory_record(base, status="read_error", message=f"parquet 读取失败：{exc}")

    missing_columns = sorted(set(CANONICAL_COLUMNS).difference(raw.columns))
    if missing_columns:
        return _inventory_record(
            base,
            status="missing_columns",
            rows=int(len(raw)),
            message=f"缺少标准行情字段：{', '.join(missing_columns)}。",
        )
    normalized = normalize_bars(raw, symbol)
    if normalized.empty:
        return _inventory_record(base, status="no_valid_rows", message="文件存在，但没有可用标准 K 线。")
    return _inventory_record(
        base,
        status="cached",
        rows=int(len(normalized)),
        start=normalized["date"].min(),
        end=normalized["date"].max(),
        message="本地 parquet 可用于读取；回测前仍建议执行覆盖率审计。",
    )


def _inventory_record(base: dict[str, object], **overrides: object) -> dict[str, object]:
    record = {
        **base,
        "status": "",
        "rows": 0,
        "start": pd.NaT,
        "end": pd.NaT,
        "message": "",
    }
    record.update(overrides)
    return record


def resolve_daily_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    if root.name.lower() == "daily":
        return root
    if root.name.lower() in set(TIMEFRAME_DIR_NAMES.values()):
        return root.parent / "daily"
    return root / "daily"


def load_local_bars(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    start_ts, end_ts = parse_time_window(start, end)
    frames: list[pd.DataFrame] = []
    for symbol in unique_symbols(tuple(symbols)):
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = normalize_bars(pd.read_parquet(path), symbol)
        frame = frame.loc[frame["date"].between(start_ts, end_ts)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return empty_bars()
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def load_daily_bars(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    root = resolve_daily_root(data_root) / adjust
    start_ts, end_ts = parse_time_window(start, end)
    frames: list[pd.DataFrame] = []
    for symbol in unique_symbols(tuple(symbols)):
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = normalize_bars(pd.read_parquet(path), symbol)
        frame = frame.loc[frame["date"].between(start_ts, end_ts)]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return empty_bars()
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def load_backtest_data(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    filter_limit_open: bool = True,
    daily_lookback_days: int = 10,
    strict_data_quality: bool = True,
    min_coverage_ratio: float | None = None,
) -> BacktestDataBundle:
    if daily_lookback_days < 1:
        raise ValueError("daily_lookback_days 至少需要 1。")
    min_coverage_ratio = _normalize_min_coverage_ratio(min_coverage_ratio)
    daily_start = pd.Timestamp(start).normalize() - pd.Timedelta(days=daily_lookback_days)
    daily = load_daily_bars(
        data_root=data_root,
        adjust=adjust,
        symbols=symbols,
        start=daily_start,
        end=end,
    )

    data_audit = audit_local_data(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
        expected_sessions_by_symbol=_daily_sessions_by_symbol(daily, start=start, end=end),
    )
    if strict_data_quality:
        _raise_for_failed_data_audit(data_audit, min_coverage_ratio=min_coverage_ratio)

    intraday = load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
    )
    intraday = _drop_zero_liquidity_bars(intraday)
    blocked = _limit_open_dates_in_window(daily, start=start, end=end) if filter_limit_open and not daily.empty else pd.DataFrame()
    filter_audit = _limit_open_filter_audit(
        daily,
        symbols=symbols,
        start=start,
        end=end,
        filter_enabled=filter_limit_open,
        blocked=blocked,
    )
    if strict_data_quality:
        _raise_for_failed_limit_filter_audit(filter_audit)
    if not filter_limit_open or intraday.empty or daily.empty:
        return BacktestDataBundle(
            bars=intraday,
            daily_bars=daily,
            filtered_limit_open_days=blocked,
            data_audit=data_audit,
            limit_filter_audit=filter_audit,
        )

    filtered = filter_limit_open_days(intraday, daily)
    return BacktestDataBundle(
        bars=filtered,
        daily_bars=daily,
        filtered_limit_open_days=blocked,
        data_audit=data_audit,
        limit_filter_audit=filter_audit,
    )


def load_multi_timeframe_backtest_data(
    *,
    data_root: str | Path,
    timeframes: tuple[str, ...] | list[str],
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    filter_limit_open: bool = True,
    daily_lookback_days: int = 10,
    strict_data_quality: bool = True,
    min_coverage_ratio: float | None = None,
) -> MultiTimeframeBacktestDataBundle:
    """一次加载多周期回测数据；每个周期独立审计，日线过滤信息统一复用。"""
    if daily_lookback_days < 1:
        raise ValueError("daily_lookback_days 至少需要 1。")
    min_coverage_ratio = _normalize_min_coverage_ratio(min_coverage_ratio)
    normalized_timeframes = _unique_timeframes(timeframes)
    if not normalized_timeframes:
        raise ValueError("timeframes 不能为空。")
    daily_start = pd.Timestamp(start).normalize() - pd.Timedelta(days=daily_lookback_days)
    daily = load_daily_bars(
        data_root=data_root,
        adjust=adjust,
        symbols=symbols,
        start=daily_start,
        end=end,
    )
    expected_sessions_by_symbol = _daily_sessions_by_symbol(daily, start=start, end=end)

    audit_frames = [
        audit_local_data(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=symbols,
            start=start,
            end=end,
            expected_sessions_by_symbol=expected_sessions_by_symbol,
        )
        for timeframe in normalized_timeframes
    ]
    data_audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame(columns=AUDIT_COLUMNS)
    if strict_data_quality:
        _raise_for_failed_data_audit(data_audit, min_coverage_ratio=min_coverage_ratio)

    blocked = _limit_open_dates_in_window(daily, start=start, end=end) if filter_limit_open and not daily.empty else pd.DataFrame()
    filter_audit = _limit_open_filter_audit(
        daily,
        symbols=symbols,
        start=start,
        end=end,
        filter_enabled=filter_limit_open,
        blocked=blocked,
    )
    if strict_data_quality:
        _raise_for_failed_limit_filter_audit(filter_audit)
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    for timeframe in normalized_timeframes:
        bars = load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=symbols,
            start=start,
            end=end,
        )
        bars = _drop_zero_liquidity_bars(bars)
        if filter_limit_open and not bars.empty and not daily.empty:
            bars = filter_limit_open_days(bars, daily)
        bars_by_timeframe[timeframe] = bars

    return MultiTimeframeBacktestDataBundle(
        bars_by_timeframe=bars_by_timeframe,
        daily_bars=daily,
        filtered_limit_open_days=blocked,
        data_audit=data_audit,
        limit_filter_audit=filter_audit,
    )


def _unique_timeframes(timeframes: tuple[str, ...] | list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in timeframes:
        timeframe = ensure_supported_timeframe(item)
        if timeframe in seen:
            continue
        seen.add(timeframe)
        result.append(timeframe)
    return result


def _drop_zero_liquidity_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """回测数据包只向 detector 暴露有真实成交的 K，零流动性数量保留在审计表。"""
    if bars.empty or "volume" not in bars.columns or "amount" not in bars.columns:
        return bars
    tradable = bars["volume"].gt(0) & bars["amount"].gt(0)
    return bars.loc[tradable].reset_index(drop=True)


def _normalize_min_coverage_ratio(value: float | None) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if pd.isna(ratio) or ratio <= 0 or ratio > 1:
        raise ValueError("min_coverage_ratio 必须在 (0, 1] 之间。")
    return ratio


def _raise_for_failed_data_audit(audit: pd.DataFrame, *, min_coverage_ratio: float | None = None) -> None:
    if audit.empty:
        return
    messages: list[str] = []
    failed = audit.loc[audit["status"] != "ok", ["stock_code", "timeframe", "status", "message"]]
    messages.extend(
        f"{row.stock_code}/{row.timeframe}={row.status}({row.message})"
        for row in failed.itertuples(index=False)
    )
    if min_coverage_ratio is not None:
        coverage_failed = audit.loc[
            (audit["expected_rows"] > 0) & (audit["coverage_ratio"] < min_coverage_ratio),
            ["stock_code", "timeframe", "coverage_ratio"],
        ]
        messages.extend(
            f"{row.stock_code}/{row.timeframe}=coverage_below_min("
            f"{_format_ratio(row.coverage_ratio)} < {_format_ratio(min_coverage_ratio)})"
            for row in coverage_failed.itertuples(index=False)
        )
    if messages:
        raise ValueError(f"本地行情数据未通过质量门禁：{'; '.join(messages)}")


def _raise_for_failed_limit_filter_audit(filter_audit: pd.DataFrame) -> None:
    """严格模式下要求日 K 过滤真实执行，避免涨停开盘日漏过滤。"""
    if filter_audit.empty:
        return
    enabled = filter_audit["filter_enabled"].astype(bool) if "filter_enabled" in filter_audit.columns else True
    failed = filter_audit.loc[
        enabled & ~filter_audit["status"].astype(str).isin({"ok"}),
        ["stock_code", "status", "message"],
    ]
    if failed.empty:
        return
    messages = [
        f"{row.stock_code}={row.status}({row.message})"
        for row in failed.itertuples(index=False)
    ]
    raise ValueError(f"日K一字涨停过滤未通过严格门禁：{'; '.join(messages)}")


def _format_ratio(value: float) -> str:
    return f"{float(value):.6g}"


def audit_local_data(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    expected_sessions_by_symbol: Mapping[str, Sequence[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """审计本地 parquet 覆盖和 OHLC 质量；回测前用它显式暴露缺口。"""
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    start_ts, end_ts = _audit_window_for_timeframe(timeframe, start=start, end=end)
    rows = [
        _audit_symbol_file(
            root=root,
            symbol=symbol,
            timeframe=timeframe,
            adjust=adjust,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_sessions=(expected_sessions_by_symbol or {}).get(symbol),
        )
        for symbol in unique_symbols(tuple(symbols))
    ]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def summarize_data_audit(audit: pd.DataFrame, *, min_coverage_ratio: float | None = None) -> dict[str, object]:
    """把行情审计表压成统计字段，便于 stats.json 和参数遍历表直接对比数据质量。"""
    if audit.empty:
        return _empty_data_audit_summary()
    min_coverage_ratio = _normalize_min_coverage_ratio(min_coverage_ratio)
    status = (
        audit["status"].fillna("").astype(str)
        if "status" in audit.columns
        else pd.Series([""] * len(audit), index=audit.index)
    )
    expected_rows = _audit_numeric_column(audit, "expected_rows")
    missing_rows = _audit_numeric_column(audit, "missing_rows")
    coverage_ratio = _audit_numeric_column(audit, "coverage_ratio")
    expected_mask = expected_rows.gt(0)
    below_min = _coverage_below_min_mask(expected_mask, coverage_ratio, min_coverage_ratio)
    expected_total = float(expected_rows.sum())
    missing_total = float(missing_rows.sum())
    max_missing_gap = _audit_numeric_column(audit, "max_missing_gap_minutes")
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
        "data_coverage_below_min_ratio": _audit_ratio_or_zero(float(below_min.sum()), float(expected_mask.sum())),
        "data_expected_rows": expected_total,
        "data_missing_rows": missing_total,
        "data_weighted_coverage_ratio": _audit_ratio_or_zero(expected_total - missing_total, expected_total),
        "data_min_coverage_ratio": _audit_min_or_zero(coverage_ratio.loc[expected_mask]),
        "data_coverage_p05": _audit_quantile_or_zero(coverage_ratio.loc[expected_mask], 0.05),
        "data_coverage_p50": _audit_quantile_or_zero(coverage_ratio.loc[expected_mask], 0.50),
        "data_coverage_p95": _audit_quantile_or_zero(coverage_ratio.loc[expected_mask], 0.95),
        "data_max_missing_gap_minutes": _audit_max_or_zero(max_missing_gap),
        **max_missing_gap_bounds,
        "data_zero_volume_amount_rows": float(_audit_numeric_column(audit, "zero_volume_amount_rows").sum()),
        "data_non_positive_price_rows": float(_audit_numeric_column(audit, "non_positive_price_rows").sum()),
        "data_negative_volume_amount_rows": float(_audit_numeric_column(audit, "negative_volume_amount_rows").sum()),
    }


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
        "data_max_missing_gap_start_at": _audit_timestamp_label(row.get("max_missing_gap_start_at")),
        "data_max_missing_gap_end_at": _audit_timestamp_label(row.get("max_missing_gap_end_at")),
    }


def _audit_timestamp_label(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


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
        "limit_filter_filtered_days": float(_audit_numeric_column(filter_audit, "filtered_days").sum()),
    }


def _audit_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def _audit_ratio_or_zero(numerator: float, denominator: float) -> float:
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


def _audit_min_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.min()), 12))


def _audit_max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.max()), 12))


def _audit_quantile_or_zero(values: pd.Series, quantile: float) -> float:
    if values.empty:
        return 0.0
    return float(round(float(values.quantile(quantile)), 12))


def _audit_window_for_timeframe(
    timeframe: str,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """日 K 审计按自然日覆盖，分钟审计保留调用方给定的盘中窗口。"""
    start_ts, end_ts = parse_time_window(start, end)
    if ensure_supported_timeframe(timeframe) == "1d":
        return start_ts.normalize(), end_ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return start_ts, end_ts


def _audit_symbol_file(
    *,
    root: Path,
    symbol: str,
    timeframe: str,
    adjust: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None = None,
) -> dict[str, object]:
    path = root / f"{symbol}.parquet"
    base = {
        "stock_code": symbol,
        "timeframe": timeframe,
        "adjust": adjust,
        "exists": path.exists(),
        "requested_start": start_ts,
        "requested_end": end_ts,
        "path": str(path),
    }
    if not path.exists():
        return _audit_record(
            base,
            status="missing_file",
            message="本地 parquet 不存在。",
        )
    try:
        raw = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return _audit_record(base, status="read_error", message=f"parquet 读取失败：{exc}")

    missing_columns = sorted(set(CANONICAL_COLUMNS).difference(raw.columns))
    if missing_columns:
        return _audit_record(
            base,
            status="missing_columns",
            rows_total=len(raw),
            missing_columns=",".join(missing_columns),
            message="缺少标准行情字段。",
        )

    checked = raw.copy()
    checked["stock_code"] = checked["stock_code"].map(normalize_symbol)
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        checked[column] = pd.to_numeric(checked[column], errors="coerce")
    invalid_date_rows = int(checked["date"].isna().sum())
    invalid_symbol_rows = int(checked["stock_code"].eq("").sum())
    window = checked.loc[checked["date"].between(start_ts, end_ts)].copy()
    duplicate_rows = int(window.duplicated(subset=["stock_code", "date"]).sum())
    null_ohlc_rows = int(window[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    non_positive_price_rows = int((window[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    inconsistent_ohlc_rows = int(_inconsistent_ohlc_mask(window).sum())
    null_volume_amount_rows = int(window[["volume", "amount"]].isna().any(axis=1).sum())
    zero_volume_amount_rows = int(window[["volume", "amount"]].eq(0).any(axis=1).sum())
    negative_volume_amount_rows = int((window[["volume", "amount"]] < 0).any(axis=1).sum())

    normalized = normalize_bars(raw, symbol)
    raw_in_window = normalized.loc[normalized["date"].between(start_ts, end_ts)]
    in_window = _drop_zero_liquidity_bars(raw_in_window)
    coverage = _intraday_session_coverage(
        in_window,
        timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
        expected_sessions=expected_sessions,
    )
    quality_error = any(
        value > 0
        for value in (
            invalid_date_rows,
            invalid_symbol_rows,
            duplicate_rows,
            null_ohlc_rows,
            non_positive_price_rows,
            inconsistent_ohlc_rows,
            null_volume_amount_rows,
            negative_volume_amount_rows,
        )
    )
    if quality_error:
        status = "quality_error"
        message = "存在日期或标的代码异常、重复时间、非法价格、OHLC 高低点不一致或量能字段异常。"
    elif in_window.empty:
        status = "no_window_data"
        message = "请求窗口内无数据。"
    elif zero_volume_amount_rows > 0:
        status = "ok"
        message = "覆盖按可交易 K 计算；存在零流动性 K，已从回测数据包剔除。"
    else:
        status = "ok"
        message = "覆盖和质量检查通过。"
    return _audit_record(
        base,
        status=status,
        rows_total=len(raw),
        rows_in_window=len(in_window),
        **coverage,
        start=in_window["date"].min() if not in_window.empty else pd.NaT,
        end=in_window["date"].max() if not in_window.empty else pd.NaT,
        invalid_date_rows=invalid_date_rows,
        invalid_symbol_rows=invalid_symbol_rows,
        duplicate_rows=duplicate_rows,
        null_ohlc_rows=null_ohlc_rows,
        non_positive_price_rows=non_positive_price_rows,
        inconsistent_ohlc_rows=inconsistent_ohlc_rows,
        null_volume_amount_rows=null_volume_amount_rows,
        zero_volume_amount_rows=zero_volume_amount_rows,
        negative_volume_amount_rows=negative_volume_amount_rows,
        message=message,
    )


def _audit_record(base: dict[str, object], **overrides: object) -> dict[str, object]:
    record = {
        **base,
        "status": "",
        "rows_total": 0,
        "rows_in_window": 0,
        "expected_rows": 0,
        "missing_rows": 0,
        "coverage_ratio": 0.0,
        "max_missing_gap_minutes": 0,
        "first_missing_at": pd.NaT,
        "last_missing_at": pd.NaT,
        "max_missing_gap_start_at": pd.NaT,
        "max_missing_gap_end_at": pd.NaT,
        "start": pd.NaT,
        "end": pd.NaT,
        "invalid_date_rows": 0,
        "invalid_symbol_rows": 0,
        "duplicate_rows": 0,
        "null_ohlc_rows": 0,
        "non_positive_price_rows": 0,
        "inconsistent_ohlc_rows": 0,
        "null_volume_amount_rows": 0,
        "zero_volume_amount_rows": 0,
        "negative_volume_amount_rows": 0,
        "missing_columns": "",
        "message": "",
    }
    record.update(overrides)
    return record


def _inconsistent_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    ohlc = frame[["open", "high", "low", "close"]]
    valid = ohlc.notna().all(axis=1) & (ohlc > 0).all(axis=1)
    max_body = ohlc[["open", "close"]].max(axis=1)
    min_body = ohlc[["open", "close"]].min(axis=1)
    high = ohlc["high"]
    low = ohlc["low"]
    return valid & ((high < max_body) | (low > min_body) | (high < low))


def _intraday_session_coverage(
    in_window: pd.DataFrame,
    timeframe: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None = None,
) -> dict[str, object]:
    if ensure_supported_timeframe(timeframe) == "1d":
        return _daily_session_coverage(
            in_window,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_sessions=expected_sessions,
        )
    if in_window.empty and not expected_sessions:
        return {
            "expected_rows": 0,
            "missing_rows": 0,
            "coverage_ratio": 0.0,
            "max_missing_gap_minutes": 0,
            "first_missing_at": pd.NaT,
            "last_missing_at": pd.NaT,
            "max_missing_gap_start_at": pd.NaT,
            "max_missing_gap_end_at": pd.NaT,
        }
    minutes = _timeframe_minutes(timeframe)
    actual_dates = pd.to_datetime(in_window["date"], errors="coerce").dropna().dt.floor("min")
    if expected_sessions:
        session_dates = pd.Series(pd.to_datetime(list(expected_sessions), errors="coerce")).dropna().dt.normalize()
        session_dates = session_dates.drop_duplicates().sort_values()
    else:
        session_dates = actual_dates.dt.normalize().drop_duplicates().sort_values()
    expected = _expected_intraday_timestamps(session_dates, minutes)
    expected = [timestamp for timestamp in expected if start_ts <= timestamp <= end_ts]
    expected_count = len(expected)
    if expected_count == 0:
        return {
            "expected_rows": 0,
            "missing_rows": 0,
            "coverage_ratio": 0.0,
            "max_missing_gap_minutes": 0,
            "first_missing_at": pd.NaT,
            "last_missing_at": pd.NaT,
            "max_missing_gap_start_at": pd.NaT,
            "max_missing_gap_end_at": pd.NaT,
        }
    actual_set = set(actual_dates)
    missing = [timestamp for timestamp in expected if timestamp not in actual_set]
    return {
        "expected_rows": int(expected_count),
        "missing_rows": int(len(missing)),
        "coverage_ratio": round((expected_count - len(missing)) / expected_count, 12),
        **_missing_coverage_summary(expected, missing=set(missing), minutes=minutes),
    }


def _daily_session_coverage(
    in_window: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None = None,
) -> dict[str, object]:
    actual_dates = pd.to_datetime(in_window["date"], errors="coerce").dropna().dt.normalize()
    if expected_sessions:
        session_dates = pd.Series(pd.to_datetime(list(expected_sessions), errors="coerce")).dropna().dt.normalize()
        session_dates = session_dates.drop_duplicates().sort_values()
    else:
        session_dates = actual_dates.drop_duplicates().sort_values()
    start_day = start_ts.normalize()
    end_day = inclusive_end_timestamp(end_ts).normalize()
    expected = [pd.Timestamp(item) for item in session_dates if start_day <= pd.Timestamp(item) <= end_day]
    expected_count = len(expected)
    if expected_count == 0:
        return {
            "expected_rows": 0,
            "missing_rows": 0,
            "coverage_ratio": 0.0,
            "max_missing_gap_minutes": 0,
            "first_missing_at": pd.NaT,
            "last_missing_at": pd.NaT,
            "max_missing_gap_start_at": pd.NaT,
            "max_missing_gap_end_at": pd.NaT,
        }
    actual_set = set(actual_dates)
    missing = [timestamp for timestamp in expected if timestamp not in actual_set]
    return {
        "expected_rows": int(expected_count),
        "missing_rows": int(len(missing)),
        "coverage_ratio": round((expected_count - len(missing)) / expected_count, 12),
        **_missing_coverage_summary(expected, missing=set(missing), minutes=1440),
    }


def _daily_sessions_by_symbol(
    daily_bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, list[pd.Timestamp]]:
    daily = normalize_bars(daily_bars)
    if daily.empty:
        return {}
    start_day = pd.Timestamp(start).normalize()
    end_day = inclusive_end_timestamp(end).normalize()
    window = daily.loc[daily["date"].dt.normalize().between(start_day, end_day)].copy()
    if window.empty:
        return {}
    window["session_date"] = window["date"].dt.normalize()
    return {
        str(symbol): sorted(group["session_date"].dropna().drop_duplicates().tolist())
        for symbol, group in window.groupby("stock_code", sort=False)
    }


def _limit_open_filter_audit(
    daily_bars: pd.DataFrame,
    *,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    filter_enabled: bool,
    blocked: pd.DataFrame,
) -> pd.DataFrame:
    daily = normalize_bars(daily_bars)
    start_day = pd.Timestamp(start).normalize()
    end_day = inclusive_end_timestamp(end).normalize()
    window = daily.loc[daily["date"].dt.normalize().between(start_day, end_day)].copy() if not daily.empty else daily
    blocked_frame = blocked.copy() if not blocked.empty else pd.DataFrame(columns=["stock_code", "session_date"])
    rows = [
        _limit_open_filter_audit_row(
            symbol=symbol,
            window=window,
            blocked=blocked_frame,
            filter_enabled=filter_enabled,
        )
        for symbol in unique_symbols(tuple(symbols))
    ]
    return pd.DataFrame(rows, columns=LIMIT_FILTER_AUDIT_COLUMNS)


def _limit_open_dates_in_window(
    daily_bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    blocked = limit_open_dates(daily_bars)
    if blocked.empty:
        return blocked
    start_day = pd.Timestamp(start).normalize()
    end_day = inclusive_end_timestamp(end).normalize()
    result = blocked.copy()
    result["session_date"] = pd.to_datetime(result["session_date"], errors="coerce").dt.normalize()
    return result.loc[result["session_date"].between(start_day, end_day)].reset_index(drop=True)


def _limit_open_filter_audit_row(
    *,
    symbol: str,
    window: pd.DataFrame,
    blocked: pd.DataFrame,
    filter_enabled: bool,
) -> dict[str, object]:
    symbol_daily = window.loc[window["stock_code"].eq(symbol)] if not window.empty else window
    filtered_days = int(blocked.loc[blocked.get("stock_code", pd.Series(dtype=str)).eq(symbol)].shape[0])
    if not filter_enabled:
        return {
            "stock_code": symbol,
            "status": "disabled",
            "filter_enabled": False,
            "daily_rows": int(len(symbol_daily)),
            "filtered_days": 0,
            "message": "日K一字涨停过滤已关闭。",
        }
    if symbol_daily.empty:
        return {
            "stock_code": symbol,
            "status": "daily_missing",
            "filter_enabled": True,
            "daily_rows": 0,
            "filtered_days": 0,
            "message": "日K缺失，无法判断一字涨停开盘过滤。",
        }
    return {
        "stock_code": symbol,
        "status": "ok",
        "filter_enabled": True,
        "daily_rows": int(len(symbol_daily)),
        "filtered_days": filtered_days,
        "message": "日K一字涨停过滤已执行。",
    }


def _timeframe_minutes(timeframe: str) -> int:
    normalized = ensure_supported_timeframe(timeframe)
    return int(normalized.removesuffix("m"))


def _expected_intraday_timestamps(session_dates: pd.Series, minutes: int) -> list[pd.Timestamp]:
    expected: list[pd.Timestamp] = []
    for session_date in session_dates:
        session = pd.Timestamp(session_date)
        expected.extend(_session_range(session, "09:30", "11:30", minutes))
        expected.extend(_session_range(session, "13:00", "15:00", minutes))
    return expected


def _session_range(session: pd.Timestamp, start: str, end: str, minutes: int) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(f"{session.date()} {start}") + pd.Timedelta(minutes=minutes)
    end_ts = pd.Timestamp(f"{session.date()} {end}")
    return [pd.Timestamp(item) for item in pd.date_range(start=start_ts, end=end_ts, freq=f"{minutes}min")]


def _missing_coverage_summary(
    expected: list[pd.Timestamp],
    missing: set[pd.Timestamp],
    minutes: int,
) -> dict[str, object]:
    """单次扫描缺失 K，返回全局缺口首尾和最长连续缺口边界。"""
    first_missing = pd.NaT
    last_missing = pd.NaT
    max_gap_minutes = 0
    max_gap_start = pd.NaT
    max_gap_end = pd.NaT
    current_gap_minutes = 0
    current_gap_start = pd.NaT
    current_gap_end = pd.NaT
    for timestamp in expected:
        if timestamp in missing:
            if pd.isna(first_missing):
                first_missing = timestamp
            last_missing = timestamp
            if current_gap_minutes == 0:
                current_gap_start = timestamp
            current_gap_minutes += minutes
            current_gap_end = timestamp
            if current_gap_minutes > max_gap_minutes:
                max_gap_minutes = current_gap_minutes
                max_gap_start = current_gap_start
                max_gap_end = current_gap_end
        else:
            current_gap_minutes = 0
            current_gap_start = pd.NaT
            current_gap_end = pd.NaT
    return {
        "max_missing_gap_minutes": int(max_gap_minutes),
        "first_missing_at": first_missing,
        "last_missing_at": last_missing,
        "max_missing_gap_start_at": max_gap_start,
        "max_missing_gap_end_at": max_gap_end,
    }


def update_from_tdx(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    timeframe: str,
    start: str,
    end: str,
    tqcenter_path: str = "",
    tq_client: Any | None = None,
) -> pd.DataFrame:
    from trending_winning.data.tdx import fetch_tdx_bars

    fetch_start, fetch_end = _tdx_fetch_window_for_timeframe(timeframe, start=start, end=end)
    bars = fetch_tdx_bars(
        symbols=symbols,
        start=fetch_start,
        end=fetch_end,
        timeframe=timeframe,
        adjust=adjust,
        tqcenter_path=tqcenter_path,
        tq_client=tq_client,
    )
    return write_local_bars(data_root=data_root, timeframe=timeframe, adjust=adjust, bars=bars)


def _tdx_fetch_window_for_timeframe(timeframe: str, *, start: str, end: str) -> tuple[str, str]:
    """日 K 请求按自然日补齐，避免分钟回测的盘中时间过滤掉日线。"""
    start_ts, end_ts = parse_time_window(start, end)
    if ensure_supported_timeframe(timeframe) != "1d":
        return start, end
    return str(start_ts.date()), str(end_ts.date())


def plan_tdx_backtest_data(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    timeframes: tuple[str, ...] | list[str],
    start: str,
    end: str,
    min_coverage_ratio: float | None = None,
) -> pd.DataFrame:
    """生成 TDX 补齐计划；只审计本地数据，不触发 TDX 请求。"""
    normalized_timeframes = _timeframes_with_daily_dependency(timeframes)
    if not normalized_timeframes:
        raise ValueError("timeframes 不能为空。")
    normalized_symbols = unique_symbols(tuple(symbols))
    min_coverage_ratio = _normalize_min_coverage_ratio(min_coverage_ratio)
    expected_sessions_by_symbol = _expected_sessions_by_symbol_from_daily(
        data_root=data_root,
        adjust=adjust,
        symbols=normalized_symbols,
        start=start,
        end=end,
    )

    rows: list[dict[str, object]] = []
    for timeframe in normalized_timeframes:
        audit = audit_local_data(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=normalized_symbols,
            start=start,
            end=end,
            expected_sessions_by_symbol=expected_sessions_by_symbol,
        )
        rows.extend(
            _tdx_plan_rows(
                audit,
                min_coverage_ratio=min_coverage_ratio,
            )
        )
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def prepare_tdx_backtest_data(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    timeframes: tuple[str, ...] | list[str],
    start: str,
    end: str,
    tqcenter_path: str = "",
    tq_client: Any | None = None,
    min_coverage_ratio: float | None = None,
    strict_after_update: bool = True,
) -> pd.DataFrame:
    """按审计结果补齐 TDX K 线；只请求缺失、坏数据或覆盖不足的标的周期。"""
    normalized_timeframes = _timeframes_with_daily_dependency(timeframes)
    if not normalized_timeframes:
        raise ValueError("timeframes 不能为空。")
    normalized_symbols = unique_symbols(tuple(symbols))
    min_coverage_ratio = _normalize_min_coverage_ratio(min_coverage_ratio)
    expected_sessions_by_symbol = _expected_sessions_by_symbol_from_daily(
        data_root=data_root,
        adjust=adjust,
        symbols=normalized_symbols,
        start=start,
        end=end,
    )
    processing_timeframes = (["1d"] if "1d" in normalized_timeframes else []) + [
        timeframe for timeframe in normalized_timeframes if timeframe != "1d"
    ]

    before_audits: dict[str, pd.DataFrame] = {}
    after_audits: dict[str, pd.DataFrame] = {}
    write_summaries: dict[str, pd.DataFrame] = {}
    fetch_symbols_by_timeframe: dict[str, list[str]] = {}

    for timeframe in processing_timeframes:
        before = audit_local_data(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=normalized_symbols,
            start=start,
            end=end,
            expected_sessions_by_symbol=expected_sessions_by_symbol,
        )
        before_audits[timeframe] = before
        fetch_symbols = [
            str(row.stock_code)
            for row in before.itertuples(index=False)
            if _audit_row_requires_tdx_update(row, min_coverage_ratio=min_coverage_ratio)
        ]
        fetch_symbols_by_timeframe[timeframe] = fetch_symbols
        if fetch_symbols:
            write_summaries[timeframe] = update_from_tdx(
                data_root=data_root,
                adjust=adjust,
                symbols=tuple(fetch_symbols),
                timeframe=timeframe,
                start=start,
                end=end,
                tqcenter_path=tqcenter_path,
                tq_client=tq_client,
            )
            after_audits[timeframe] = audit_local_data(
                data_root=data_root,
                timeframe=timeframe,
                adjust=adjust,
                symbols=normalized_symbols,
                start=start,
                end=end,
                expected_sessions_by_symbol=expected_sessions_by_symbol,
            )
        else:
            write_summaries[timeframe] = pd.DataFrame()
            after_audits[timeframe] = before
        if timeframe == "1d":
            expected_sessions_by_symbol = _expected_sessions_by_symbol_from_daily(
                data_root=data_root,
                adjust=adjust,
                symbols=normalized_symbols,
                start=start,
                end=end,
            )

    after_all = pd.concat(after_audits.values(), ignore_index=True) if after_audits else pd.DataFrame(columns=AUDIT_COLUMNS)
    if strict_after_update:
        _raise_for_failed_data_audit(after_all, min_coverage_ratio=min_coverage_ratio)

    rows: list[dict[str, object]] = []
    for timeframe in normalized_timeframes:
        rows.extend(
            _prepare_summary_rows(
                before=before_audits[timeframe],
                after=after_audits[timeframe],
                write_summary=write_summaries[timeframe],
                fetched_symbols=set(fetch_symbols_by_timeframe[timeframe]),
                min_coverage_ratio=min_coverage_ratio,
            )
        )
    return pd.DataFrame(rows, columns=PREPARE_COLUMNS)


def _timeframes_with_daily_dependency(timeframes: tuple[str, ...] | list[str]) -> list[str]:
    """TDX 回测数据准备自动带上日 K；日 K 是分钟覆盖锚点和涨停开盘过滤依赖。"""
    normalized = _unique_timeframes(timeframes)
    if not normalized:
        return normalized
    if any(timeframe != "1d" for timeframe in normalized) and "1d" not in normalized:
        return ["1d", *normalized]
    return normalized


def _expected_sessions_by_symbol_from_daily(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, list[pd.Timestamp]]:
    start_day = pd.Timestamp(start).normalize()
    end_day = pd.Timestamp(end).normalize()
    daily = load_daily_bars(
        data_root=data_root,
        adjust=adjust,
        symbols=tuple(symbols),
        start=start_day,
        end=end_day,
    )
    return _daily_sessions_by_symbol(daily, start=start, end=end)


def _tdx_plan_rows(audit: pd.DataFrame, *, min_coverage_ratio: float | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in audit.itertuples(index=False):
        requires_fetch = _audit_row_requires_tdx_update(row, min_coverage_ratio=min_coverage_ratio)
        reason = _summary_before_status(row, min_coverage_ratio=min_coverage_ratio) if requires_fetch else "local_ok"
        rows.append(
            {
                "stock_code": str(row.stock_code),
                "timeframe": str(row.timeframe),
                "adjust": str(row.adjust),
                "action": "fetch" if requires_fetch else "cached",
                "reason": reason,
                "before_status": _summary_before_status(row, min_coverage_ratio=min_coverage_ratio),
                "rows_in_window": int(row.rows_in_window),
                "expected_rows": int(row.expected_rows),
                "missing_rows": int(row.missing_rows),
                "coverage_ratio": float(row.coverage_ratio),
                "max_missing_gap_minutes": int(row.max_missing_gap_minutes),
                "first_missing_at": getattr(row, "first_missing_at"),
                "last_missing_at": getattr(row, "last_missing_at"),
                "max_missing_gap_start_at": getattr(row, "max_missing_gap_start_at"),
                "max_missing_gap_end_at": getattr(row, "max_missing_gap_end_at"),
                "path": str(row.path),
                "message": str(row.message),
            }
        )
    return rows


def _audit_row_requires_tdx_update(row: object, *, min_coverage_ratio: float | None) -> bool:
    status = str(getattr(row, "status"))
    if status != "ok":
        return True
    if min_coverage_ratio is None:
        return False
    expected_rows = int(getattr(row, "expected_rows"))
    coverage_ratio = float(getattr(row, "coverage_ratio"))
    return expected_rows > 0 and coverage_ratio < min_coverage_ratio


def _summary_before_status(row: object, *, min_coverage_ratio: float | None) -> str:
    if _audit_row_requires_tdx_update(row, min_coverage_ratio=min_coverage_ratio) and str(getattr(row, "status")) == "ok":
        return "coverage_below_min"
    return str(getattr(row, "status"))


def _prepare_summary_rows(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    write_summary: pd.DataFrame,
    fetched_symbols: set[str],
    min_coverage_ratio: float | None,
) -> list[dict[str, object]]:
    after_by_symbol = {str(row.stock_code): row for row in after.itertuples(index=False)}
    write_by_symbol = {
        str(row.symbol): row
        for row in write_summary.itertuples(index=False)
    } if not write_summary.empty else {}

    rows: list[dict[str, object]] = []
    for before_row in before.itertuples(index=False):
        symbol = str(before_row.stock_code)
        after_row = after_by_symbol.get(symbol, before_row)
        write_row = write_by_symbol.get(symbol)
        rows.append(
            {
                "stock_code": symbol,
                "timeframe": str(before_row.timeframe),
                "adjust": str(before_row.adjust),
                "action": "fetched" if symbol in fetched_symbols else "cached",
                "before_status": _summary_before_status(before_row, min_coverage_ratio=min_coverage_ratio),
                "after_status": str(after_row.status),
                "rows_written": int(getattr(write_row, "rows", 0)) if write_row is not None else 0,
                "new_rows": int(getattr(write_row, "new_rows", 0)) if write_row is not None else 0,
                "before_coverage_ratio": float(before_row.coverage_ratio),
                "after_coverage_ratio": float(after_row.coverage_ratio),
                "coverage_ratio": float(after_row.coverage_ratio),
                "before_missing_rows": int(before_row.missing_rows),
                "after_missing_rows": int(after_row.missing_rows),
                "missing_rows": int(after_row.missing_rows),
                "before_max_missing_gap_minutes": int(before_row.max_missing_gap_minutes),
                "after_max_missing_gap_minutes": int(after_row.max_missing_gap_minutes),
                "before_first_missing_at": getattr(before_row, "first_missing_at"),
                "before_last_missing_at": getattr(before_row, "last_missing_at"),
                "after_first_missing_at": getattr(after_row, "first_missing_at"),
                "after_last_missing_at": getattr(after_row, "last_missing_at"),
                "first_missing_at": getattr(after_row, "first_missing_at"),
                "last_missing_at": getattr(after_row, "last_missing_at"),
                "before_max_missing_gap_start_at": getattr(before_row, "max_missing_gap_start_at"),
                "before_max_missing_gap_end_at": getattr(before_row, "max_missing_gap_end_at"),
                "after_max_missing_gap_start_at": getattr(after_row, "max_missing_gap_start_at"),
                "after_max_missing_gap_end_at": getattr(after_row, "max_missing_gap_end_at"),
                "max_missing_gap_start_at": getattr(after_row, "max_missing_gap_start_at"),
                "max_missing_gap_end_at": getattr(after_row, "max_missing_gap_end_at"),
                "path": str(after_row.path),
                "message": str(after_row.message),
            }
        )
    return rows


def write_local_bars(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    bars: pd.DataFrame,
) -> pd.DataFrame:
    normalized = normalize_bars(bars)
    if normalized.empty:
        return empty_download_result()

    root = resolve_timeframe_root(data_root, timeframe) / adjust
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for symbol, incoming in normalized.groupby("stock_code", sort=True):
        path = root / f"{symbol}.parquet"
        previous_rows = 0
        if path.exists():
            previous = normalize_bars(pd.read_parquet(path), str(symbol))
            previous_rows = len(previous)
            merged = pd.concat([previous, incoming], ignore_index=True)
        else:
            merged = incoming.copy()
        merged = (
            merged[CANONICAL_COLUMNS]
            .drop_duplicates(subset=["stock_code", "date"], keep="last")
            .sort_values(["stock_code", "date"])
            .reset_index(drop=True)
        )
        merged.to_parquet(path, index=False)
        rows.append(
            {
                "symbol": str(symbol),
                "status": "success",
                "rows": int(len(merged)),
                "new_rows": int(max(len(merged) - previous_rows, 0)),
                "path": str(path),
                "start": merged["date"].min(),
                "end": merged["date"].max(),
                "message": "TDX 行情已写入本地 parquet。",
            }
        )
    return pd.DataFrame(rows)
