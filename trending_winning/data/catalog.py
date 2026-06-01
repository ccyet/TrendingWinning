from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sqlite3

import pandas as pd

from trending_winning.data.schema import TIMEFRAME_DIR_NAMES, normalize_symbol

CATALOG_FILE_NAME = "market_data_catalog.sqlite"
CATALOG_COLUMNS = [
    "cache_key",
    "stock_code",
    "stock_name",
    "asset_type",
    "data_kind",
    "indicator",
    "timeframe",
    "adjust",
    "storage_format",
    "status",
    "rows",
    "start_at",
    "end_at",
    "file_size_bytes",
    "modified_at",
    "path",
    "message",
]
ASSET_TYPE_LABELS = {
    "stock": "个股",
    "index": "指数",
    "etf": "ETF",
    "other": "其他",
}
DATA_KIND_LABELS = {"price": "价格成交", "indicator": "技术指标"}
INDICATOR_LABELS = {"ohlcv": "原始OHLCV"}


def catalog_path_for(data_root: str | Path) -> Path:
    root = _catalog_root(data_root)
    return root / "metadata" / CATALOG_FILE_NAME


def enrich_inventory_for_catalog(inventory: pd.DataFrame, *, symbol_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = inventory.copy()
    if frame.empty:
        return pd.DataFrame(columns=pd.Index(CATALOG_COLUMNS))
    name_by_symbol = _symbol_name_map(symbol_metadata)
    frame["stock_code"] = frame["stock_code"].map(normalize_symbol)
    frame["stock_name"] = frame["stock_code"].map(lambda symbol: name_by_symbol.get(symbol, ""))
    frame["asset_type"] = [
        infer_asset_type(symbol, name)
        for symbol, name in zip(frame["stock_code"], frame["stock_name"], strict=False)
    ]
    frame["data_kind"] = "price"
    frame["indicator"] = "ohlcv"
    frame["storage_format"] = "parquet"
    frame["start_at"] = frame.get("start", pd.Series([pd.NaT] * len(frame))).map(_timestamp_text)
    frame["end_at"] = frame.get("end", pd.Series([pd.NaT] * len(frame))).map(_timestamp_text)
    frame["modified_at"] = frame.get("modified_at", pd.Series([pd.NaT] * len(frame))).map(_timestamp_text)
    frame["cache_key"] = [
        "|".join(
            [
                str(row.stock_code),
                str(row.asset_type),
                str(row.data_kind),
                str(row.indicator),
                str(row.timeframe),
                str(row.adjust),
                str(row.path),
            ]
        )
        for row in frame.itertuples(index=False)
    ]
    for column in CATALOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame.loc[:, CATALOG_COLUMNS].reset_index(drop=True)


def build_catalog(
    *,
    data_root: str | Path,
    inventory: pd.DataFrame,
    symbol_metadata: pd.DataFrame | None = None,
) -> Path:
    path = catalog_path_for(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = enrich_inventory_for_catalog(inventory, symbol_metadata=symbol_metadata)
    with sqlite3.connect(path) as connection:
        _init_catalog(connection)
        connection.execute("DELETE FROM market_data_files")
        if not records.empty:
            records.to_sql("market_data_files", connection, if_exists="append", index=False)
        connection.commit()
    return path


def query_catalog(
    *,
    data_root: str | Path,
    asset_types: Sequence[str] | None = None,
    timeframes: Sequence[str] | None = None,
    indicators: Sequence[str] | None = None,
    data_kinds: Sequence[str] | None = None,
    statuses: Sequence[str] | None = None,
) -> pd.DataFrame:
    path = catalog_path_for(data_root)
    if not path.exists():
        return pd.DataFrame(columns=pd.Index(CATALOG_COLUMNS))
    where: list[str] = []
    params: list[object] = []
    for column, values in (
        ("asset_type", asset_types),
        ("timeframe", timeframes),
        ("indicator", indicators),
        ("data_kind", data_kinds),
        ("status", statuses),
    ):
        values_tuple = tuple(str(value) for value in values or () if str(value))
        if not values_tuple:
            continue
        placeholders = ",".join("?" for _ in values_tuple)
        where.append(f"{column} IN ({placeholders})")
        params.extend(values_tuple)
    sql = "SELECT * FROM market_data_files"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY asset_type, timeframe, stock_code"
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def infer_asset_type(symbol: object, stock_name: object = "") -> str:
    normalized = normalize_symbol(symbol)
    if not normalized:
        return "other"
    code, exchange = normalized.split(".", 1)
    name = str(stock_name or "").upper()
    if "ETF" in name or "LOF" in name or "基金" in name or _looks_like_etf_code(code):
        return "etf"
    if (exchange == "SH" and code.startswith(("000", "880"))) or (exchange == "SZ" and code.startswith("399")):
        return "index"
    if code.startswith(("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689")):
        return "stock"
    if exchange == "BJ" and code.startswith(("430", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "920")):
        return "stock"
    return "other"


def asset_type_label(value: object) -> str:
    return ASSET_TYPE_LABELS.get(str(value), str(value))


def data_kind_label(value: object) -> str:
    return DATA_KIND_LABELS.get(str(value), str(value))


def indicator_label(value: object) -> str:
    return INDICATOR_LABELS.get(str(value), str(value))


def _init_catalog(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data_files (
            cache_key TEXT PRIMARY KEY,
            stock_code TEXT NOT NULL,
            stock_name TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            data_kind TEXT NOT NULL,
            indicator TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            adjust TEXT NOT NULL,
            storage_format TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL,
            start_at TEXT NOT NULL,
            end_at TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            path TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_data_lookup "
        "ON market_data_files(asset_type, data_kind, indicator, timeframe, adjust, stock_code, status)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_data_window "
        "ON market_data_files(stock_code, timeframe, start_at, end_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_market_data_status "
        "ON market_data_files(status, asset_type, timeframe)"
    )


def _catalog_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    if root.name.lower() in set(TIMEFRAME_DIR_NAMES.values()):
        return root.parent
    return root


def _symbol_name_map(symbol_metadata: pd.DataFrame | None) -> dict[str, str]:
    if symbol_metadata is None or symbol_metadata.empty:
        return {}
    return {
        normalize_symbol(row.stock_code): str(row.stock_name)
        for row in symbol_metadata.itertuples(index=False)
        if normalize_symbol(row.stock_code)
    }


def _timestamp_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat()


def _looks_like_etf_code(code: str) -> bool:
    return code.startswith(("159", "510", "511", "512", "513", "515", "516", "517", "518", "520", "560", "561", "562", "563", "588"))
