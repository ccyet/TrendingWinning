from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from trending_winning.data.filters import limit_open_dates
from trending_winning.data.schema import (
    CANONICAL_COLUMNS,
    ensure_supported_timeframe,
    inclusive_end_timestamp,
    normalize_bars,
    normalize_symbol,
    parse_time_window,
    resolve_timeframe_root,
    unique_symbols,
)


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

DATA_GAP_EPISODE_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "gap_no",
    "start_at",
    "end_at",
    "missing_rows",
    "gap_minutes",
    "previous_available_at",
    "next_available_at",
    "requested_start",
    "requested_end",
    "path",
    "status",
]

LIMIT_FILTER_AUDIT_COLUMNS = [
    "stock_code",
    "status",
    "filter_enabled",
    "daily_rows",
    "filtered_days",
    "message",
]

DAILY_FILTER_AUDIT_STATUS_BY_DATA_STATUS = {
    "read_error": "daily_read_error",
    "missing_columns": "daily_missing_columns",
    "quality_error": "daily_quality_error",
}
DAILY_FILTER_AUDIT_MESSAGE_BY_DATA_STATUS = {
    "read_error": "日K parquet 读取失败",
    "missing_columns": "日K parquet 缺少标准字段",
    "quality_error": "日K parquet 存在质量异常",
}


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


def data_gap_episodes(
    *,
    data_root: str | Path,
    timeframe: str,
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    expected_sessions_by_symbol: Mapping[str, Sequence[pd.Timestamp]] | None = None,
) -> pd.DataFrame:
    """输出连续缺失 K 段，便于用户直接定位要补哪一段缓存。"""
    normalized_timeframe = ensure_supported_timeframe(timeframe)
    root = resolve_timeframe_root(data_root, normalized_timeframe) / adjust
    start_ts, end_ts = _audit_window_for_timeframe(normalized_timeframe, start=start, end=end)
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(tuple(symbols)):
        rows.extend(
            _symbol_gap_episodes(
                root=root,
                symbol=symbol,
                timeframe=normalized_timeframe,
                adjust=adjust,
                start_ts=start_ts,
                end_ts=end_ts,
                expected_sessions=(expected_sessions_by_symbol or {}).get(symbol),
            )
        )
    return pd.DataFrame(rows, columns=DATA_GAP_EPISODE_COLUMNS)


def daily_sessions_by_symbol(
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


def limit_open_filter_audit(
    daily_bars: pd.DataFrame,
    *,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    filter_enabled: bool,
    blocked: pd.DataFrame,
    daily_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    daily = normalize_bars(daily_bars)
    start_day = pd.Timestamp(start).normalize()
    end_day = inclusive_end_timestamp(end).normalize()
    window = daily.loc[daily["date"].dt.normalize().between(start_day, end_day)].copy() if not daily.empty else daily
    blocked_frame = blocked.copy() if not blocked.empty else pd.DataFrame(columns=["stock_code", "session_date"])
    audit_by_symbol = _daily_audit_by_symbol(daily_audit)
    rows = [
        _limit_open_filter_audit_row(
            symbol=symbol,
            window=window,
            blocked=blocked_frame,
            filter_enabled=filter_enabled,
            daily_audit_row=audit_by_symbol.get(symbol),
        )
        for symbol in unique_symbols(tuple(symbols))
    ]
    return pd.DataFrame(rows, columns=LIMIT_FILTER_AUDIT_COLUMNS)


def limit_open_dates_in_window(
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
        return _audit_record(base, status="missing_file", message="本地 parquet 不存在。")
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
    window = checked.loc[checked["date"].between(start_ts, end_ts)].copy()
    invalid_symbol_rows = int(window["stock_code"].eq("").sum())
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


def _symbol_gap_episodes(
    *,
    root: Path,
    symbol: str,
    timeframe: str,
    adjust: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None = None,
) -> list[dict[str, object]]:
    path = root / f"{symbol}.parquet"
    base = {
        "stock_code": symbol,
        "timeframe": timeframe,
        "adjust": adjust,
        "requested_start": start_ts,
        "requested_end": end_ts,
        "path": str(path),
    }
    if not path.exists():
        return _gap_episodes_for_dates(
            pd.DataFrame(columns=["date"]),
            timeframe,
            expected_sessions=expected_sessions,
            start_ts=start_ts,
            end_ts=end_ts,
            base=base,
            status="missing_file",
        )
    try:
        raw = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return []
    if sorted(set(CANONICAL_COLUMNS).difference(raw.columns)):
        return _gap_episodes_for_dates(
            pd.DataFrame(columns=["date"]),
            timeframe,
            expected_sessions=expected_sessions,
            start_ts=start_ts,
            end_ts=end_ts,
            base=base,
            status="missing_columns",
        )

    normalized = normalize_bars(raw, symbol)
    raw_in_window = normalized.loc[normalized["date"].between(start_ts, end_ts)]
    in_window = _drop_zero_liquidity_bars(raw_in_window)
    return _gap_episodes_for_dates(
        in_window,
        timeframe,
        expected_sessions=expected_sessions,
        start_ts=start_ts,
        end_ts=end_ts,
        base=base,
        status="missing_bars",
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


def _drop_zero_liquidity_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or "volume" not in bars.columns or "amount" not in bars.columns:
        return bars
    tradable = bars["volume"].gt(0) & bars["amount"].gt(0)
    return bars.loc[tradable].reset_index(drop=True)


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
        return _empty_coverage()
    minutes = _timeframe_minutes(timeframe)
    expected, actual_set = _intraday_expected_and_actual(
        in_window,
        minutes,
        start_ts=start_ts,
        end_ts=end_ts,
        expected_sessions=expected_sessions,
    )
    expected_count = len(expected)
    if expected_count == 0:
        return _empty_coverage()
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
    expected, actual_set = _daily_expected_and_actual(
        in_window,
        start_ts=start_ts,
        end_ts=end_ts,
        expected_sessions=expected_sessions,
    )
    expected_count = len(expected)
    if expected_count == 0:
        return _empty_coverage()
    missing = [timestamp for timestamp in expected if timestamp not in actual_set]
    return {
        "expected_rows": int(expected_count),
        "missing_rows": int(len(missing)),
        "coverage_ratio": round((expected_count - len(missing)) / expected_count, 12),
        **_missing_coverage_summary(expected, missing=set(missing), minutes=1440),
    }


def _empty_coverage() -> dict[str, object]:
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


def _gap_episodes_for_dates(
    in_window: pd.DataFrame,
    timeframe: str,
    *,
    expected_sessions: Sequence[pd.Timestamp] | None,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    base: dict[str, object],
    status: str,
) -> list[dict[str, object]]:
    if ensure_supported_timeframe(timeframe) == "1d":
        expected, actual_set = _daily_expected_and_actual(
            in_window,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_sessions=expected_sessions,
        )
        minutes = 1440
    else:
        minutes = _timeframe_minutes(timeframe)
        expected, actual_set = _intraday_expected_and_actual(
            in_window,
            minutes,
            start_ts=start_ts,
            end_ts=end_ts,
            expected_sessions=expected_sessions,
        )
    return _missing_gap_episode_rows(expected, actual_set=actual_set, minutes=minutes, base=base, status=status)


def _intraday_expected_and_actual(
    in_window: pd.DataFrame,
    minutes: int,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None,
) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    actual_dates = _actual_minute_dates(in_window)
    if expected_sessions:
        session_dates = pd.Series(pd.to_datetime(list(expected_sessions), errors="coerce")).dropna().dt.normalize()
        session_dates = session_dates.drop_duplicates().sort_values()
    else:
        session_dates = actual_dates.dt.normalize().drop_duplicates().sort_values()
    expected = _expected_intraday_timestamps(session_dates, minutes)
    expected = [timestamp for timestamp in expected if start_ts <= timestamp <= end_ts]
    return expected, set(actual_dates)


def _daily_expected_and_actual(
    in_window: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    expected_sessions: Sequence[pd.Timestamp] | None,
) -> tuple[list[pd.Timestamp], set[pd.Timestamp]]:
    actual_dates = _actual_minute_dates(in_window).dt.normalize()
    if expected_sessions:
        session_dates = pd.Series(pd.to_datetime(list(expected_sessions), errors="coerce")).dropna().dt.normalize()
        session_dates = session_dates.drop_duplicates().sort_values()
    else:
        session_dates = actual_dates.drop_duplicates().sort_values()
    start_day = start_ts.normalize()
    end_day = inclusive_end_timestamp(end_ts).normalize()
    expected = [pd.Timestamp(item) for item in session_dates if start_day <= pd.Timestamp(item) <= end_day]
    return expected, set(actual_dates)


def _actual_minute_dates(in_window: pd.DataFrame) -> pd.Series:
    if in_window.empty or "date" not in in_window.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(in_window["date"], errors="coerce").dropna().dt.floor("min")


def _missing_gap_episode_rows(
    expected: list[pd.Timestamp],
    *,
    actual_set: set[pd.Timestamp],
    minutes: int,
    base: dict[str, object],
    status: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous_available = pd.NaT
    gap_start = pd.NaT
    gap_end = pd.NaT
    gap_previous = pd.NaT
    missing_rows = 0
    for timestamp in expected:
        if timestamp not in actual_set:
            if missing_rows == 0:
                gap_start = timestamp
                gap_previous = previous_available
            missing_rows += 1
            gap_end = timestamp
            continue
        if missing_rows > 0:
            rows.append(
                _gap_episode_row(
                    base,
                    gap_no=len(rows) + 1,
                    start_at=gap_start,
                    end_at=gap_end,
                    missing_rows=missing_rows,
                    minutes=minutes,
                    previous_available_at=gap_previous,
                    next_available_at=timestamp,
                    status=status,
                )
            )
            missing_rows = 0
            gap_start = pd.NaT
            gap_end = pd.NaT
            gap_previous = pd.NaT
        previous_available = timestamp
    if missing_rows > 0:
        rows.append(
            _gap_episode_row(
                base,
                gap_no=len(rows) + 1,
                start_at=gap_start,
                end_at=gap_end,
                missing_rows=missing_rows,
                minutes=minutes,
                previous_available_at=gap_previous,
                next_available_at=pd.NaT,
                status=status,
            )
        )
    return rows


def _gap_episode_row(
    base: dict[str, object],
    *,
    gap_no: int,
    start_at: pd.Timestamp,
    end_at: pd.Timestamp,
    missing_rows: int,
    minutes: int,
    previous_available_at: pd.Timestamp,
    next_available_at: pd.Timestamp,
    status: str,
) -> dict[str, object]:
    return {
        **base,
        "gap_no": int(gap_no),
        "start_at": start_at,
        "end_at": end_at,
        "missing_rows": int(missing_rows),
        "gap_minutes": int(missing_rows * minutes),
        "previous_available_at": previous_available_at,
        "next_available_at": next_available_at,
        "status": status,
    }


def _daily_audit_by_symbol(daily_audit: pd.DataFrame | None) -> dict[str, Mapping[str, object]]:
    if daily_audit is None or daily_audit.empty or "stock_code" not in daily_audit.columns:
        return {}
    result: dict[str, Mapping[str, object]] = {}
    for row in daily_audit.to_dict("records"):
        symbol = normalize_symbol(row.get("stock_code", ""))
        if symbol:
            result[symbol] = row
    return result


def _limit_open_filter_audit_row(
    *,
    symbol: str,
    window: pd.DataFrame,
    blocked: pd.DataFrame,
    filter_enabled: bool,
    daily_audit_row: Mapping[str, object] | None = None,
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
        audit_status = str(daily_audit_row.get("status", "")) if daily_audit_row is not None else ""
        status = DAILY_FILTER_AUDIT_STATUS_BY_DATA_STATUS.get(audit_status)
        if status is not None:
            detail = str(daily_audit_row.get("message", "")) if daily_audit_row is not None else ""
            summary = DAILY_FILTER_AUDIT_MESSAGE_BY_DATA_STATUS[audit_status]
            return {
                "stock_code": symbol,
                "status": status,
                "filter_enabled": True,
                "daily_rows": 0,
                "filtered_days": 0,
                "message": f"{summary}，无法判断一字涨停开盘过滤：{detail}",
            }
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
