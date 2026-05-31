from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from trending_winning.data.schema import (
    CANONICAL_COLUMNS,
    SUPPORTED_TIMEFRAMES,
    ensure_supported_timeframe,
    normalize_symbol,
    resolve_timeframe_root,
    unique_symbols,
)


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
    "missing_columns",
    "path",
    "message",
]


def available_symbols(data_root: str | Path, timeframe: str, adjust: str = "qfq") -> list[str]:
    root = resolve_timeframe_root(data_root, timeframe) / adjust
    if not root.exists():
        return []
    return _parquet_file_symbols(root)


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
        for symbol in _parquet_file_symbols(root):
            if symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return sorted(symbols)


def _parquet_file_symbols(root: Path) -> list[str]:
    return sorted({symbol for path in root.glob("*.parquet") if path.is_file() and (symbol := normalize_symbol(path.stem))})


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
        parquet_file = pq.ParquetFile(path)
    except Exception as exc:  # noqa: BLE001
        return _inventory_record(base, status="read_error", message=f"parquet 元数据读取失败：{exc}")

    missing_columns = sorted(set(CANONICAL_COLUMNS).difference(parquet_file.schema.names))
    if missing_columns:
        return _inventory_record(
            base,
            status="missing_columns",
            rows=_parquet_num_rows(parquet_file),
            missing_columns=",".join(missing_columns),
            message=f"缺少标准行情字段：{', '.join(missing_columns)}。",
        )
    try:
        identity = pd.read_parquet(path, columns=["date", "stock_code"])
    except Exception as exc:  # noqa: BLE001
        return _inventory_record(base, status="read_error", message=f"parquet 关键列读取失败：{exc}")
    valid_identity = _valid_inventory_identity(identity)
    if valid_identity.empty:
        return _inventory_record(base, status="no_valid_rows", message="文件存在，但没有可用标准 K 线。")
    return _inventory_record(
        base,
        status="cached",
        rows=int(len(valid_identity)),
        start=valid_identity["date"].min(),
        end=valid_identity["date"].max(),
        message="本地 parquet 可用于读取；回测前仍建议执行覆盖率审计。",
    )


def _parquet_num_rows(parquet_file: pq.ParquetFile) -> int:
    metadata = parquet_file.metadata
    return int(metadata.num_rows) if metadata is not None else 0


def _valid_inventory_identity(identity: pd.DataFrame) -> pd.DataFrame:
    """库存扫描只需要代码和日期；完整 OHLC 质量交给 audit-data。"""
    result = identity.loc[:, ["date", "stock_code"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["stock_code"] = result["stock_code"].map(normalize_symbol).replace("", pd.NA)
    result = result.dropna(subset=["date", "stock_code"])
    return result.drop_duplicates(subset=["stock_code", "date"], keep="last").sort_values(
        ["stock_code", "date"],
        kind="mergesort",
    )


def _inventory_record(base: dict[str, object], **overrides: object) -> dict[str, object]:
    record = {
        **base,
        "status": "",
        "rows": 0,
        "start": pd.NaT,
        "end": pd.NaT,
        "missing_columns": "",
        "message": "",
    }
    record.update(overrides)
    return record
