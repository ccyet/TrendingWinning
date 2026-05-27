from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_TIMEFRAMES = ("1d", "5m", "15m", "30m", "60m")
TIMEFRAME_DIR_NAMES = {"1d": "daily", **{timeframe: timeframe for timeframe in SUPPORTED_TIMEFRAMES if timeframe != "1d"}}
CANONICAL_COLUMNS = ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]


def ensure_supported_timeframe(timeframe: str) -> str:
    normalized = str(timeframe).strip().lower()
    if normalized not in SUPPORTED_TIMEFRAMES:
        supported = "、".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(f"timeframe 仅支持 {supported}。")
    return normalized


def normalize_symbol(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    text = str(value).strip().upper().replace("_", ".")
    if not text:
        return ""
    if "." in text:
        code, exchange = text.split(".", 1)
        digits = "".join(character for character in code if character.isdigit())
        exchange_code = exchange[:2]
        if not digits or exchange_code not in {"SH", "SZ", "BJ"}:
            return ""
        return f"{digits.zfill(6)}.{exchange_code}"

    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return ""
    code = digits[-6:].zfill(6)
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{code}.SH"
    if code.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{code}.SZ"
    if code.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "920")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def unique_symbols(symbols: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in symbols:
        symbol = normalize_symbol(item)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def inclusive_end_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if isinstance(value, str):
        text = value.strip()
        if " " not in text and "T" not in text and len(text) <= 10:
            return timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return timestamp


def parse_time_window(start: str | pd.Timestamp, end: str | pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """解析并校验时间窗口；配置错误必须显式失败，不能静默返回空数据。"""
    start_ts = pd.Timestamp(start)
    end_ts = inclusive_end_timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start 不能晚于 end。")
    return start_ts, end_ts


def resolve_timeframe_root(data_root: str | Path, timeframe: str) -> Path:
    normalized = ensure_supported_timeframe(timeframe)
    root = Path(data_root).expanduser()
    target_dir = TIMEFRAME_DIR_NAMES[normalized]
    known_dirs = set(TIMEFRAME_DIR_NAMES.values())
    root_name = root.name.lower()
    if root_name == target_dir:
        return root
    if root_name in known_dirs:
        return root.parent / target_dir
    return root / target_dir


def normalize_bars(frame: pd.DataFrame, fallback_symbol: str = "") -> pd.DataFrame:
    result = frame.copy()
    if "stock_code" not in result.columns and "symbol" in result.columns:
        result = result.rename(columns={"symbol": "stock_code"})
    if "stock_code" not in result.columns:
        result["stock_code"] = normalize_symbol(fallback_symbol)
    result["stock_code"] = result["stock_code"].map(normalize_symbol)
    result["stock_code"] = result["stock_code"].replace("", pd.NA)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in result.columns:
            result[column] = pd.NA
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["date", "stock_code", "open", "high", "low", "close"])
    result = result[CANONICAL_COLUMNS].drop_duplicates(subset=["stock_code", "date"], keep="last")
    return result.sort_values(["stock_code", "date"]).reset_index(drop=True)


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(CANONICAL_COLUMNS))


def empty_download_result() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "status", "rows", "new_rows", "path", "start", "end", "message"])


def first_present(payload: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        if key in payload:
            return key
    return None
