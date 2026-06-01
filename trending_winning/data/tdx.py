from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from trending_winning.data.schema import (
    CANONICAL_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    ensure_supported_timeframe,
    empty_bars,
    first_present,
    normalize_symbol,
    parse_time_window,
    unique_symbols,
)

TDX_TQCENTER_ENV_VAR = "TDX_TQCENTER_PATH"
TDX_ALLOW_MAC_TQCENTER_ENV_VAR = "TDX_ALLOW_MAC_TQCENTER"
TDX_REQUEST_BATCH_SIZE = 100
REFRESHABLE_KLINE_PERIODS = {"1m", "5m"}
TDX_PERIOD_MAP = {"1d": "1d", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h"}
DERIVABLE_FROM_5M_TIMEFRAMES = {"15m", "30m", "60m"}
ADJUST_MAP = {"": "none", "qfq": "front", "hfq": "back"}
TDX_DIAG_COLUMNS = ["timeframe", "tdx_period", "status", "rows", "symbols", "start", "end", "message"]
REQUIRED_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Amount")
FIELD_ALIASES = {
    "Open": ("Open", "open"),
    "High": ("High", "high"),
    "Low": ("Low", "low"),
    "Close": ("Close", "close"),
    "Volume": ("Volume", "volume", "vol"),
    "Amount": ("Amount", "amount"),
}
OUTPUT_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Amount": "amount",
}

_TQ_CLIENT: Any | None = None
_INITIALIZED_CLIENT_ID: int | None = None
_INITIALIZED_CLIENT: Any | None = None
ProgressCallback = Callable[[dict[str, object]], None]


def fetch_tdx_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframe: str,
    adjust: str = "qfq",
    tqcenter_path: str = "",
    tq_client: Any | None = None,
    batch_size: int = TDX_REQUEST_BATCH_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    normalized_timeframe = ensure_supported_timeframe(timeframe)
    normalized_batch_size = _normalize_batch_size(batch_size)
    period = TDX_PERIOD_MAP[normalized_timeframe]
    dividend_type = ADJUST_MAP.get(str(adjust))
    if dividend_type is None:
        raise ValueError("adjust 仅支持 qfq、hfq 或空字符串。")
    parse_time_window(start, end)

    normalized_symbols = unique_symbols(tuple(symbols))
    if not normalized_symbols:
        return empty_bars()

    _emit_progress(
        progress_callback,
        stage="tdx_request_start",
        timeframe=normalized_timeframe,
        period=period,
        symbol_count=len(normalized_symbols),
    )
    tq = tq_client or _load_tq(tqcenter_path)
    _ensure_initialized(tq)

    bars = _fetch_tdx_period_bars(
        tq,
        symbols=normalized_symbols,
        start=start,
        end=end,
        period=period,
        dividend_type=dividend_type,
        batch_size=normalized_batch_size,
        progress_callback=progress_callback,
        timeframe=normalized_timeframe,
    )
    if normalized_timeframe not in DERIVABLE_FROM_5M_TIMEFRAMES:
        _emit_progress(
            progress_callback,
            stage="tdx_request_done",
            timeframe=normalized_timeframe,
            period=period,
            rows=len(bars),
        )
        return bars
    missing_symbols = _symbols_missing_bars(normalized_symbols, bars)
    if not missing_symbols:
        _emit_progress(
            progress_callback,
            stage="tdx_request_done",
            timeframe=normalized_timeframe,
            period=period,
            rows=len(bars),
        )
        return bars

    fallback_start, fallback_end = _expanded_5m_aggregation_window(start, end)
    _emit_progress(
        progress_callback,
        stage="tdx_fallback_start",
        timeframe=normalized_timeframe,
        period="5m",
        symbol_count=len(missing_symbols),
    )
    five_minute_bars = _fetch_tdx_period_bars(
        tq,
        symbols=missing_symbols,
        start=fallback_start,
        end=fallback_end,
        period="5m",
        dividend_type=dividend_type,
        batch_size=normalized_batch_size,
        progress_callback=progress_callback,
        timeframe=normalized_timeframe,
    )
    derived = _aggregate_5m_bars(
        five_minute_bars,
        timeframe=normalized_timeframe,
        start=start,
        end=end,
    )
    if bars.empty:
        result = derived
    elif derived.empty:
        result = bars
    else:
        result = pd.concat([bars, derived], ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)
    _emit_progress(
        progress_callback,
        stage="tdx_request_done",
        timeframe=normalized_timeframe,
        period=period,
        rows=len(result),
    )
    return result


def diagnose_tdx_source(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframes: tuple[str, ...] | list[str] = SUPPORTED_TIMEFRAMES,
    adjust: str = "qfq",
    tqcenter_path: str = "",
    tq_client: Any | None = None,
) -> pd.DataFrame:
    """诊断 TDX 通道；逐周期返回样本请求状态、行数和明确错误。"""
    normalized_symbols = unique_symbols(tuple(symbols))
    normalized_timeframes = [ensure_supported_timeframe(timeframe) for timeframe in timeframes]
    parse_time_window(start, end)
    if not normalized_symbols:
        return pd.DataFrame(
            [
                _diagnosis_row(
                    timeframe=timeframe,
                    status="invalid_symbols",
                    message="没有可识别的 A 股代码。",
                )
                for timeframe in normalized_timeframes
            ],
            columns=TDX_DIAG_COLUMNS,
        )

    try:
        tq = tq_client or _load_tq(tqcenter_path)
        _ensure_initialized(tq)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(
            [
                _diagnosis_row(
                    timeframe=timeframe,
                    status="init_error",
                    message=str(exc),
                    symbols=normalized_symbols,
                )
                for timeframe in normalized_timeframes
            ],
            columns=TDX_DIAG_COLUMNS,
        )

    rows: list[dict[str, object]] = []
    for timeframe in normalized_timeframes:
        request_start, request_end = _diagnosis_request_window(timeframe, start, end)
        try:
            bars = fetch_tdx_bars(
                symbols=tuple(normalized_symbols),
                start=request_start,
                end=request_end,
                timeframe=timeframe,
                adjust=adjust,
                tq_client=tq,
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                _diagnosis_row(
                    timeframe=timeframe,
                    status="request_error",
                    message=str(exc),
                    symbols=normalized_symbols,
                )
            )
            continue
        if bars.empty:
            rows.append(
                _diagnosis_row(
                    timeframe=timeframe,
                    status="no_data",
                    message=_no_data_diagnosis_message(timeframe),
                    symbols=normalized_symbols,
                )
            )
            continue
        rows.append(
            _diagnosis_row(
                timeframe=timeframe,
                status="ok",
                rows=len(bars),
                symbols=sorted(bars["stock_code"].dropna().astype(str).unique().tolist()),
                start=bars["date"].min(),
                end=bars["date"].max(),
                message="TDX 样本请求成功。",
            )
        )
    return pd.DataFrame(rows, columns=TDX_DIAG_COLUMNS)


def _diagnosis_request_window(timeframe: str, start: str, end: str) -> tuple[str, str]:
    """日 K 诊断按整天取样，避免分钟起点把当天 00:00 日线排除。"""
    if timeframe != "1d":
        return start, end
    return str(pd.Timestamp(start).normalize().date()), str(pd.Timestamp(end).normalize().date())


def _no_data_diagnosis_message(timeframe: str) -> str:
    """把 TDX 空结果翻译成可执行排障结论，避免误判为 Mac 本机取数。"""
    if ensure_supported_timeframe(timeframe) == "1d":
        return "TDX 请求成功但样本窗口无 K 线。"
    return (
        "TDX 请求成功但样本窗口无分钟 K 线；"
        "Parallels/Windows 通达信本地没有返回分钟 K 线，Mac 本机通达信不参与取数。"
        "请先在 Windows 通达信内确认 5m 分钟数据已下载，15m/30m/60m 可由原生周期或 5m 聚合生成。"
    )


def _fetch_tdx_period_bars(
    tq: Any,
    *,
    symbols: list[str],
    start: str,
    end: str,
    period: str,
    dividend_type: str,
    batch_size: int,
    progress_callback: ProgressCallback | None,
    timeframe: str,
) -> pd.DataFrame:
    """按一个 TDX 原生周期请求并标准化；高周期 fallback 复用这个低层入口。"""
    frames: list[pd.DataFrame] = []
    batches = _batched_symbols(symbols, batch_size)
    for batch_index, batch in enumerate(batches, start=1):
        _emit_progress(
            progress_callback,
            stage="tdx_batch_start",
            timeframe=timeframe,
            period=period,
            batch_index=batch_index,
            batch_count=len(batches),
            symbol_count=len(batch),
        )
        _refresh_tdx_kline_cache(tq, batch, period)
        payload = tq.get_market_data(
            field_list=list(REQUIRED_FIELDS),
            stock_list=batch,
            period=period,
            start_time=_format_market_time(start),
            end_time=_format_market_time(end),
            count=-1,
            dividend_type=dividend_type,
            fill_data=False,
        )
        batch_frames: list[pd.DataFrame] = []
        for symbol in batch:
            frame = _normalize_tdx_payload(payload, symbol=symbol, start=start, end=end)
            if not frame.empty:
                batch_frames.append(frame)
        frames.extend(batch_frames)
        _emit_progress(
            progress_callback,
            stage="tdx_batch_done",
            timeframe=timeframe,
            period=period,
            batch_index=batch_index,
            batch_count=len(batches),
            symbol_count=len(batch),
            rows=sum(len(frame) for frame in batch_frames),
        )
    if not frames:
        return empty_bars()
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


def _expanded_5m_aggregation_window(start: str, end: str) -> tuple[str, str]:
    """高周期由 5m 聚合时按整段交易日取数，避免起点裁掉首根高周期 K 的组成 5m。"""
    start_ts, end_ts = parse_time_window(start, end)
    return f"{start_ts.date()} 09:30:00", f"{end_ts.date()} 15:00:00"


def _symbols_missing_bars(symbols: list[str], bars: pd.DataFrame) -> list[str]:
    if bars.empty or "stock_code" not in bars.columns:
        return list(symbols)
    present = set(bars["stock_code"].dropna().astype(str))
    return [symbol for symbol in symbols if symbol not in present]


def _aggregate_5m_bars(bars: pd.DataFrame, *, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """把 TDX 5m K 线聚合成 15/30/60m；只保留组成数量完整的目标 K。"""
    if bars.empty:
        return empty_bars()
    minutes = int(ensure_supported_timeframe(timeframe).removesuffix("m"))
    if minutes <= 5 or minutes % 5 != 0:
        raise ValueError("5m 聚合只支持 15m、30m、60m。")
    data = bars.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["_bucket_end"] = _intraday_bucket_ends(data["date"], minutes)
    data = data.dropna(subset=["date", "_bucket_end"])
    if data.empty:
        return empty_bars()
    data = data.sort_values(["stock_code", "date"])
    grouped = (
        data.groupby(["stock_code", "_bucket_end"], sort=True, dropna=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            bar_count=("date", "count"),
        )
        .reset_index()
        .rename(columns={"_bucket_end": "date"})
    )
    complete_count = minutes // 5
    grouped = grouped.loc[grouped["bar_count"].ge(complete_count)].copy()
    if grouped.empty:
        return empty_bars()
    start_ts, end_ts = _filter_window(start, end)
    grouped = grouped.loc[grouped["date"].between(start_ts, end_ts)].copy()
    if grouped.empty:
        return empty_bars()
    return grouped[CANONICAL_COLUMNS].sort_values(["stock_code", "date"]).reset_index(drop=True)


def _intraday_bucket_ends(values: pd.Series, minutes: int) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    normalized_dates = timestamps.dt.normalize()
    for session_start_text, session_end_text in (("09:30", "11:30"), ("13:00", "15:00")):
        start_hour, start_minute = (int(part) for part in session_start_text.split(":"))
        end_hour, end_minute = (int(part) for part in session_end_text.split(":"))
        session_start = normalized_dates + pd.Timedelta(hours=start_hour, minutes=start_minute)
        session_end = normalized_dates + pd.Timedelta(hours=end_hour, minutes=end_minute)
        in_session = timestamps.gt(session_start) & timestamps.le(session_end)
        if not bool(in_session.any()):
            continue
        elapsed_minutes = ((timestamps.loc[in_session] - session_start.loc[in_session]).dt.total_seconds() // 60).astype(int)
        bucket_minutes = ((elapsed_minutes + minutes - 1) // minutes) * minutes
        bucket_end = session_start.loc[in_session] + pd.to_timedelta(bucket_minutes, unit="m")
        result.loc[in_session] = bucket_end.where(bucket_end.le(session_end.loc[in_session]), pd.NaT)
    return result


def _diagnosis_row(
    *,
    timeframe: str,
    status: str,
    message: str,
    rows: int = 0,
    symbols: list[str] | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> dict[str, object]:
    return {
        "timeframe": timeframe,
        "tdx_period": TDX_PERIOD_MAP.get(timeframe, ""),
        "status": status,
        "rows": int(rows),
        "symbols": ",".join(symbols or []),
        "start": pd.Timestamp(start) if start is not None else pd.NaT,
        "end": pd.Timestamp(end) if end is not None else pd.NaT,
        "message": message,
    }


def _load_tq(tqcenter_path: str = "") -> Any:
    global _TQ_CLIENT
    if _mac_local_tq_disabled():
        raise RuntimeError(
            "Mac 通达信不支持 tqcenter 取数。请在 Mac 端使用 CLI 参数 `--runtime parallels` "
            "调用 Parallels/Windows 通达信，或直接在 Windows 侧运行本项目。"
        )
    if _TQ_CLIENT is not None:
        return _TQ_CLIENT

    errors: list[str] = []
    for path in _candidate_import_paths(tqcenter_path):
        resolved = path.resolve()
        if not resolved.exists():
            errors.append(f"{resolved} 不存在")
            continue
        path_text = str(resolved)
        inserted = False
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted = True
        try:
            module = importlib.import_module("tqcenter")
            _TQ_CLIENT = getattr(module, "tq")
            return _TQ_CLIENT
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path_text}: {exc}")
            if inserted:
                try:
                    sys.path.remove(path_text)
                except ValueError:
                    pass

    try:
        module = importlib.import_module("tqcenter")
        _TQ_CLIENT = getattr(module, "tq")
        return _TQ_CLIENT
    except Exception as exc:  # noqa: BLE001
        errors.append(f"normal import: {exc}")
        details = " | ".join(errors)
        raise RuntimeError(
            "无法导入 tqcenter。请先在 Windows/Parallels 内启动并登录通达信终端，并通过 "
            f"{TDX_TQCENTER_ENV_VAR} 或页面文件夹选择器指向 PYPlugins/user。详情: {details}"
        ) from exc


def _mac_local_tq_disabled() -> bool:
    return sys.platform == "darwin" and os.getenv(TDX_ALLOW_MAC_TQCENTER_ENV_VAR, "") != "1"


def _candidate_import_paths(tqcenter_path: str = "") -> list[Path]:
    raw_value = (tqcenter_path or os.getenv(TDX_TQCENTER_ENV_VAR, "")).strip()
    if not raw_value or raw_value.lower() == "tdx":
        return []

    def expand(candidate: Path) -> list[Path]:
        normalized = candidate
        if normalized.name.lower() == "tqcenter.py":
            normalized = normalized.parent
        if normalized.name.lower() == "user" and normalized.parent.name.lower() == "pyplugins":
            return [normalized]
        if normalized.name.lower() == "pyplugins":
            return [normalized / "user", normalized]
        return [normalized / "PYPlugins" / "user", normalized]

    paths: list[Path] = []
    seen: set[str] = set()
    for item in raw_value.split(os.pathsep):
        text = item.strip().strip('"')
        if not text:
            continue
        for path in expand(Path(text).expanduser()):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
    return paths


def _ensure_initialized(tq: Any) -> None:
    global _INITIALIZED_CLIENT
    if _INITIALIZED_CLIENT is tq:
        return
    try:
        tq.initialize(__file__)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("TDX 初始化失败。请确认本机通达信终端已启动并登录。") from exc
    _INITIALIZED_CLIENT = tq


def _refresh_tdx_kline_cache(tq: Any, symbols: list[str], period: str) -> None:
    if period not in REFRESHABLE_KLINE_PERIODS:
        return
    refresh = getattr(tq, "refresh_kline", None)
    if not callable(refresh):
        return
    try:
        result = refresh(list(symbols), period)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"TDX K线缓存刷新失败：{exc}") from exc
    error = _tdx_refresh_error(result)
    if error:
        raise RuntimeError(f"TDX K线缓存刷新失败：{error}")


def _tdx_refresh_error(result: object) -> str:
    if result is None:
        return "接口无返回"
    if isinstance(result, Mapping):
        payload = result
    elif isinstance(result, str):
        text = result.strip()
        if not text:
            return "接口返回为空"
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
    else:
        return ""
    error_id = str(payload.get("ErrorId", "0"))
    if error_id in {"", "0", "None"}:
        return ""
    return str(payload.get("Error") or payload.get("Msg") or payload)


def _normalize_tdx_payload(raw_data: Any, *, symbol: str, start: str, end: str) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(symbol)
    if raw_data is None:
        return empty_bars()
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"TDX 返回应为 dict[field]->DataFrame，实际为 {type(raw_data).__name__}。")
    if not raw_data:
        return empty_bars()
    if "error" in raw_data and "msg" in raw_data:
        raise ValueError(f"TDX 返回错误：{raw_data.get('msg')}")

    selected_frames: dict[str, pd.DataFrame] = {}
    symbol_presence: list[bool] = []
    for field in REQUIRED_FIELDS:
        key = first_present(dict(raw_data), FIELD_ALIASES[field])
        if key is None:
            raise ValueError(f"TDX 返回缺少必要字段：{field}")
        value = raw_data[key]
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"TDX 字段 {key} 应为 DataFrame，实际为 {type(value).__name__}。")
        columns = {str(column): column for column in value.columns}
        symbol_presence.append(normalized_symbol in columns)
        selected_frames[field] = value
    if not any(symbol_presence):
        return empty_bars()
    if not all(symbol_presence):
        missing_fields = [
            field
            for field, present in zip(REQUIRED_FIELDS, symbol_presence, strict=True)
            if not present
        ]
        raise ValueError(f"TDX 标的 {normalized_symbol} 缺少部分字段列：{', '.join(missing_fields)}")

    if all(frame.empty for frame in selected_frames.values()):
        return empty_bars()

    series_map: dict[str, pd.Series] = {}
    for field, frame in selected_frames.items():
        column = {str(item): item for item in frame.columns}[normalized_symbol]
        series_map[field] = pd.Series(
            pd.to_numeric(frame[column].to_numpy(), errors="coerce"),
            index=pd.to_datetime(frame.index, errors="coerce"),
            name=field,
        )

    assembled = pd.concat(series_map, axis=1).reset_index().rename(columns={"index": "date"})
    assembled = assembled.rename(columns=OUTPUT_RENAME)
    assembled["date"] = pd.to_datetime(assembled["date"], errors="coerce")
    start_ts, end_ts = _filter_window(start, end)
    assembled = assembled.loc[assembled["date"].between(start_ts, end_ts)].copy()
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        assembled[column] = pd.to_numeric(assembled[column], errors="coerce")
    assembled["stock_code"] = normalized_symbol
    assembled = assembled.dropna(subset=["date", "open", "high", "low", "close"])
    assembled = assembled[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    return assembled.sort_values("date").reset_index(drop=True)


def _batched_symbols(symbols: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError("batch_size 至少需要 1。")
    return [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]


def _normalize_batch_size(value: int) -> int:
    batch_size = int(value)
    if batch_size < 1:
        raise ValueError("TDX 请求批次大小至少需要 1。")
    return batch_size


def _emit_progress(callback: ProgressCallback | None, **payload: object) -> None:
    if callback is not None:
        callback(payload)


def _format_market_time(value: str) -> str:
    parsed = pd.Timestamp(pd.to_datetime(value))
    return parsed.strftime("%Y%m%d%H%M%S" if _has_explicit_time(value) else "%Y%m%d")


def _has_explicit_time(value: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", str(value).strip()))


def _filter_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return parse_time_window(start, end)
