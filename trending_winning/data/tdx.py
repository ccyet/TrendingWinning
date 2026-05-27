from __future__ import annotations

from collections.abc import Mapping
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
TDX_REQUEST_BATCH_SIZE = 100
REFRESHABLE_KLINE_PERIODS = {"5m"}
TDX_PERIOD_MAP = {"1d": "1d", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h"}
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


def fetch_tdx_bars(
    *,
    symbols: tuple[str, ...] | list[str],
    start: str,
    end: str,
    timeframe: str,
    adjust: str = "qfq",
    tqcenter_path: str = "",
    tq_client: Any | None = None,
) -> pd.DataFrame:
    normalized_timeframe = ensure_supported_timeframe(timeframe)
    period = TDX_PERIOD_MAP[normalized_timeframe]
    dividend_type = ADJUST_MAP.get(str(adjust))
    if dividend_type is None:
        raise ValueError("adjust 仅支持 qfq、hfq 或空字符串。")
    parse_time_window(start, end)

    normalized_symbols = unique_symbols(tuple(symbols))
    if not normalized_symbols:
        return empty_bars()

    tq = tq_client or _load_tq(tqcenter_path)
    _ensure_initialized(tq)

    frames: list[pd.DataFrame] = []
    for batch in _batched_symbols(normalized_symbols, TDX_REQUEST_BATCH_SIZE):
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
        for symbol in batch:
            frame = _normalize_tdx_payload(payload, symbol=symbol, start=start, end=end)
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return empty_bars()
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


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
                    message="TDX 请求成功但样本窗口无 K 线。",
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
            "无法导入 tqcenter。请先安装并登录本机通达信终端，并通过 "
            f"{TDX_TQCENTER_ENV_VAR} 或页面输入框指向 PYPlugins/user。详情: {details}"
        ) from exc


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
    for field in REQUIRED_FIELDS:
        key = first_present(dict(raw_data), FIELD_ALIASES[field])
        if key is None:
            raise ValueError(f"TDX 返回缺少必要字段：{field}")
        value = raw_data[key]
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"TDX 字段 {key} 应为 DataFrame，实际为 {type(value).__name__}。")
        columns = {str(column): column for column in value.columns}
        if normalized_symbol not in columns:
            raise ValueError(f"TDX 字段 {key} 缺少标的列：{normalized_symbol}")
        selected_frames[field] = value

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


def _format_market_time(value: str) -> str:
    parsed = pd.Timestamp(pd.to_datetime(value))
    return parsed.strftime("%Y%m%d%H%M%S" if _has_explicit_time(value) else "%Y%m%d")


def _has_explicit_time(value: str) -> bool:
    return bool(re.search(r"\d{1,2}:\d{2}", str(value).strip()))


def _filter_window(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return parse_time_window(start, end)
