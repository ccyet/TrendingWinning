from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from trending_winning.data.audit import (
    AUDIT_COLUMNS,
    LIMIT_FILTER_AUDIT_COLUMNS,
    audit_local_data,
    daily_sessions_by_symbol,
    limit_open_dates_in_window,
    limit_open_filter_audit,
)
from trending_winning.data.filters import filter_limit_open_days
from trending_winning.data.inventory import (
    INVENTORY_COLUMNS,
    available_symbols,
    inventory_local_data,
)
from trending_winning.data.schema import (
    SUPPORTED_TIMEFRAMES,
    ensure_supported_timeframe,
    normalize_symbol,
    parse_time_window,
    resolve_timeframe_root,
    unique_symbols,
)
from trending_winning.data.summary import (
    DATA_AUDIT_SUMMARY_KEYS,
    DATA_INVENTORY_SUMMARY_KEYS,
    LIMIT_FILTER_SUMMARY_KEYS,
    normalize_min_coverage_ratio as _normalize_min_coverage_ratio,
    summarize_data_audit,
    summarize_data_inventory,
    summarize_data_management,
    summarize_limit_filter_audit,
)
from trending_winning.data.storage import load_daily_bars, load_local_bars, resolve_daily_root, write_local_bars
from trending_winning.data.symbols import load_symbol_metadata, resolve_symbol_names

__all__ = [
    "BacktestDataBundle",
    "DATA_AUDIT_SUMMARY_KEYS",
    "DATA_INVENTORY_SUMMARY_KEYS",
    "INVENTORY_COLUMNS",
    "LIMIT_FILTER_SUMMARY_KEYS",
    "MultiTimeframeBacktestDataBundle",
    "MarketDataRepository",
    "audit_local_data",
    "available_symbols",
    "inventory_local_data",
    "load_backtest_data",
    "load_daily_bars",
    "load_local_bars",
    "load_multi_timeframe_backtest_data",
    "load_symbol_metadata",
    "plan_tdx_backtest_data",
    "prepare_tdx_backtest_data",
    "resolve_daily_root",
    "resolve_symbol_names",
    "resolve_timeframe_root",
    "summarize_data_audit",
    "summarize_data_inventory",
    "summarize_data_management",
    "summarize_limit_filter_audit",
    "update_from_tdx",
    "write_local_bars",
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

UNLOADABLE_AUDIT_STATUSES = frozenset({"read_error", "missing_columns"})
DAILY_DEPENDENCY_FAILURE_STATUSES = frozenset({"read_error", "missing_columns", "quality_error"})


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

    def symbol_metadata(self, *, tdx_path: str | Path = "") -> pd.DataFrame:
        """返回股票代码和名称元数据；供 UI、统计展示和导出解释复用。"""
        return load_symbol_metadata(self.data_root, tdx_path=tdx_path)

    def symbol_names(
        self,
        *,
        symbols: tuple[str, ...] | list[str],
        tdx_path: str | Path = "",
    ) -> dict[str, str]:
        """返回指定股票的名称映射；本地 sidecar/TDX 优先，常用代码兜底。"""
        return resolve_symbol_names(symbols, data_root=self.data_root, tdx_path=tdx_path)

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
    daily, daily_audit = _load_daily_dependency_for_backtest(
        data_root=data_root,
        adjust=adjust,
        symbols=symbols,
        start=daily_start,
        end=end,
        strict_data_quality=strict_data_quality,
    )

    data_audit = audit_local_data(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
        expected_sessions_by_symbol=daily_sessions_by_symbol(daily, start=start, end=end),
    )
    if strict_data_quality:
        _raise_for_failed_data_audit(data_audit, min_coverage_ratio=min_coverage_ratio)

    intraday_symbols = _symbols_safe_for_backtest_load(
        symbols=symbols,
        audit=data_audit,
        timeframe=timeframe,
        strict_data_quality=strict_data_quality,
    )
    intraday = load_local_bars(
        data_root=data_root,
        timeframe=timeframe,
        adjust=adjust,
        symbols=intraday_symbols,
        start=start,
        end=end,
    )
    intraday = _drop_zero_liquidity_bars(intraday)
    blocked = limit_open_dates_in_window(daily, start=start, end=end) if filter_limit_open and not daily.empty else pd.DataFrame()
    filter_audit = limit_open_filter_audit(
        daily,
        symbols=symbols,
        start=start,
        end=end,
        filter_enabled=filter_limit_open,
        blocked=blocked,
        daily_audit=daily_audit,
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
    daily, daily_audit = _load_daily_dependency_for_backtest(
        data_root=data_root,
        adjust=adjust,
        symbols=symbols,
        start=daily_start,
        end=end,
        strict_data_quality=strict_data_quality,
    )
    expected_sessions_by_symbol = daily_sessions_by_symbol(daily, start=start, end=end)

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

    blocked = limit_open_dates_in_window(daily, start=start, end=end) if filter_limit_open and not daily.empty else pd.DataFrame()
    filter_audit = limit_open_filter_audit(
        daily,
        symbols=symbols,
        start=start,
        end=end,
        filter_enabled=filter_limit_open,
        blocked=blocked,
        daily_audit=daily_audit,
    )
    if strict_data_quality:
        _raise_for_failed_limit_filter_audit(filter_audit)
    bars_by_timeframe: dict[str, pd.DataFrame] = {}
    for timeframe in normalized_timeframes:
        timeframe_symbols = _symbols_safe_for_backtest_load(
            symbols=symbols,
            audit=data_audit,
            timeframe=timeframe,
            strict_data_quality=strict_data_quality,
        )
        bars = load_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust=adjust,
            symbols=timeframe_symbols,
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


def _load_daily_dependency_for_backtest(
    *,
    data_root: str | Path,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    strict_data_quality: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先审计再读取日 K 依赖；损坏文件不能绕过审计直接让回测崩溃。"""
    daily_audit = audit_local_data(
        data_root=data_root,
        timeframe="1d",
        adjust=adjust,
        symbols=symbols,
        start=start,
        end=end,
    )
    if strict_data_quality:
        hard_failures = daily_audit.loc[
            daily_audit["status"].astype(str).isin(DAILY_DEPENDENCY_FAILURE_STATUSES)
        ]
        _raise_for_failed_data_audit(hard_failures)

    daily_symbols = _symbols_safe_for_backtest_load(
        symbols=symbols,
        audit=daily_audit,
        timeframe="1d",
        strict_data_quality=strict_data_quality,
    )
    daily = load_daily_bars(
        data_root=data_root,
        adjust=adjust,
        symbols=daily_symbols,
        start=start,
        end=end,
    )
    return daily, daily_audit


def _symbols_safe_for_backtest_load(
    *,
    symbols: tuple[str, ...] | list[str],
    audit: pd.DataFrame,
    timeframe: str,
    strict_data_quality: bool,
) -> list[str]:
    """非严格回测只跳过无法读取的文件；质量问题仍交给 normalize_bars 暴露到结果中。"""
    normalized_symbols = unique_symbols(tuple(symbols))
    if strict_data_quality or audit.empty or not {"stock_code", "timeframe", "status"}.issubset(audit.columns):
        return normalized_symbols
    unsafe_rows = audit.loc[
        audit["timeframe"].astype(str).eq(timeframe)
        & audit["status"].astype(str).isin(UNLOADABLE_AUDIT_STATUSES),
        "stock_code",
    ]
    unsafe_symbols = {normalize_symbol(symbol) for symbol in unsafe_rows}
    return [symbol for symbol in normalized_symbols if symbol not in unsafe_symbols]


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
    return daily_sessions_by_symbol(daily, start=start, end=end)


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
