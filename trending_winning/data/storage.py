from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from trending_winning.data.schema import (
    CANONICAL_COLUMNS,
    TIMEFRAME_DIR_NAMES,
    empty_bars,
    empty_download_result,
    normalize_bars,
    parse_time_window,
    resolve_timeframe_root,
    unique_symbols,
)


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
        frame = _read_bars_parquet_window(path, symbol=symbol, start_ts=start_ts, end_ts=end_ts)
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
        frame = _read_bars_parquet_window(path, symbol=symbol, start_ts=start_ts, end_ts=end_ts)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return empty_bars()
    return pd.concat(frames, ignore_index=True).sort_values(["stock_code", "date"]).reset_index(drop=True)


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


def _read_bars_parquet_window(
    path: Path,
    *,
    symbol: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """按标准列和时间窗口读取 parquet，减少分钟线大文件的无效 IO。"""
    try:
        frame = pd.read_parquet(
            path,
            columns=list(CANONICAL_COLUMNS),
            filters=_date_window_filters(start_ts, end_ts),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
        if not _is_date_filter_type_mismatch(exc):
            raise
        frame = pd.read_parquet(path, columns=list(CANONICAL_COLUMNS))
    normalized = normalize_bars(frame, symbol)
    return normalized.loc[normalized["date"].between(start_ts, end_ts)].reset_index(drop=True)


def _date_window_filters(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[str, str, pd.Timestamp]]:
    return [("date", ">=", start_ts), ("date", "<=", end_ts)]


def _is_date_filter_type_mismatch(exc: Exception) -> bool:
    message = str(exc).lower()
    return "timestamp" in message and "string" in message and ("greater_equal" in message or "less_equal" in message)
