from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trending_winning.data.catalog import (
    ASSET_TYPE_LABELS,
    build_catalog,
    catalog_path_for,
    enrich_inventory_for_catalog,
    query_catalog,
)
from trending_winning.data.repository import MarketDataRepository
from trending_winning.data.schema import SUPPORTED_TIMEFRAMES, ensure_supported_timeframe, unique_symbols
from trending_winning.data.summary import summarize_data_inventory

ProgressCallback = Callable[[dict[str, object]], None]

DOWNLOAD_MODES = ("smart", "force")
QUICK_SYMBOL_GROUPS = {
    "核心样例": ("000001.SZ", "600519.SH", "300750.SZ", "601318.SH"),
    "宽基指数": ("000001.SH", "399001.SZ", "399006.SZ", "000300.SH", "000852.SH", "000905.SH"),
    "ETF样例": ("510300.SH", "510500.SH", "159915.SZ", "588000.SH", "512100.SH"),
}
FORCE_DOWNLOAD_COLUMNS = [
    "stock_code",
    "timeframe",
    "adjust",
    "action",
    "rows_written",
    "new_rows",
    "start",
    "end",
    "path",
    "message",
]


@dataclass(frozen=True)
class DataDownloadConfig:
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    start: str
    end: str
    tqcenter_path: str = ""
    batch_size: int = 100
    min_coverage_ratio: float | None = None
    strict_after_update: bool = True


@dataclass(frozen=True)
class DataCacheSnapshot:
    inventory: pd.DataFrame
    catalog: pd.DataFrame
    catalog_path: Path
    summary: dict[str, object]
    readiness: pd.DataFrame
    by_timeframe: pd.DataFrame
    by_status: pd.DataFrame
    by_asset_type: pd.DataFrame
    by_dataset: pd.DataFrame


@dataclass(frozen=True)
class DataDownloadResult:
    table: pd.DataFrame
    summary: dict[str, object]


class DataManagementService:
    """独立数据管理服务；只依赖行情数据层，不依赖回测和 Streamlit。"""

    def __init__(self, data_root: str | Path, adjust: str = "qfq") -> None:
        self.data_root = Path(data_root).expanduser()
        self.adjust = adjust
        self.repository = MarketDataRepository(self.data_root, adjust=adjust)

    def cache_snapshot(
        self,
        *,
        timeframes: tuple[str, ...] | list[str] = SUPPORTED_TIMEFRAMES,
        symbols: tuple[str, ...] | list[str] | None = None,
        asset_types: tuple[str, ...] | list[str] | None = None,
        tdx_path: str | Path = "",
        rebuild_catalog: bool = True,
    ) -> DataCacheSnapshot:
        normalized_timeframes = normalize_timeframes(timeframes)
        normalized_symbols = normalize_symbol_tuple(symbols) if symbols is not None else None
        inventory = self.repository.inventory(timeframes=normalized_timeframes, symbols=normalized_symbols)
        metadata = self.repository.symbol_metadata(tdx_path=tdx_path)
        catalog = enrich_inventory_for_catalog(inventory, symbol_metadata=metadata)
        catalog_path = catalog_path_for(self.data_root)
        if rebuild_catalog:
            catalog_path = build_catalog(data_root=self.data_root, inventory=inventory, symbol_metadata=metadata)
        if asset_types:
            allowed = {str(item) for item in asset_types}
            catalog = catalog.loc[catalog["asset_type"].isin(allowed)].reset_index(drop=True)
            inventory = inventory.loc[inventory["stock_code"].isin(catalog["stock_code"])].reset_index(drop=True)
        return DataCacheSnapshot(
            inventory=inventory,
            catalog=catalog,
            catalog_path=catalog_path,
            summary=cache_summary(catalog),
            readiness=cache_readiness(catalog),
            by_timeframe=cache_by_timeframe(catalog),
            by_status=cache_by_status(catalog),
            by_asset_type=cache_by_asset_type(catalog),
            by_dataset=cache_by_dataset(catalog),
        )

    def catalog_query(
        self,
        *,
        asset_types: tuple[str, ...] | list[str] | None = None,
        timeframes: tuple[str, ...] | list[str] | None = None,
        indicators: tuple[str, ...] | list[str] | None = None,
        statuses: tuple[str, ...] | list[str] | None = None,
    ) -> pd.DataFrame:
        return query_catalog(
            data_root=self.data_root,
            asset_types=asset_types,
            timeframes=timeframes,
            indicators=indicators,
            statuses=statuses,
        )

    def cached_symbols(
        self,
        *,
        asset_types: tuple[str, ...] | list[str] | None = None,
        timeframes: tuple[str, ...] | list[str] | None = None,
        tdx_path: str | Path = "",
    ) -> tuple[str, ...]:
        snapshot = self.cache_snapshot(
            timeframes=tuple(timeframes or SUPPORTED_TIMEFRAMES),
            symbols=None,
            asset_types=asset_types,
            tdx_path=tdx_path,
            rebuild_catalog=False,
        )
        if snapshot.catalog.empty:
            return ()
        return tuple(sorted(snapshot.catalog["stock_code"].dropna().astype(str).unique().tolist()))

    def download_plan(self, config: DataDownloadConfig) -> pd.DataFrame:
        return self.repository.plan_from_tdx(
            symbols=normalize_symbol_tuple(config.symbols),
            timeframes=normalize_timeframes(config.timeframes),
            start=config.start,
            end=config.end,
            min_coverage_ratio=config.min_coverage_ratio,
        )

    def download(
        self,
        config: DataDownloadConfig,
        *,
        mode: str = "smart",
        progress_callback: ProgressCallback | None = None,
    ) -> DataDownloadResult:
        normalized_mode = normalize_download_mode(mode)
        if normalized_mode == "smart":
            table = self.repository.prepare_from_tdx(
                symbols=normalize_symbol_tuple(config.symbols),
                timeframes=normalize_timeframes(config.timeframes),
                start=config.start,
                end=config.end,
                tqcenter_path=config.tqcenter_path,
                batch_size=config.batch_size,
                progress_callback=progress_callback,
                min_coverage_ratio=config.min_coverage_ratio,
                strict_after_update=config.strict_after_update,
            )
        else:
            table = self._force_download(config, progress_callback=progress_callback)
        return DataDownloadResult(table=table, summary=download_summary(table))

    def _force_download(
        self,
        config: DataDownloadConfig,
        *,
        progress_callback: ProgressCallback | None,
    ) -> pd.DataFrame:
        symbols = normalize_symbol_tuple(config.symbols)
        timeframes = normalize_timeframes(config.timeframes)
        frames: list[pd.DataFrame] = []
        for step_index, timeframe in enumerate(timeframes, start=1):
            _emit_progress(
                progress_callback,
                stage="force_timeframe_start",
                timeframe=timeframe,
                step_index=step_index,
                step_count=len(timeframes),
            )
            written = self.repository.update_from_tdx(
                symbols=symbols,
                timeframe=timeframe,
                start=config.start,
                end=config.end,
                tqcenter_path=config.tqcenter_path,
                batch_size=config.batch_size,
                progress_callback=progress_callback,
            )
            frames.append(_force_download_frame(written, timeframe=timeframe, adjust=self.adjust))
            _emit_progress(
                progress_callback,
                stage="force_timeframe_done",
                timeframe=timeframe,
                step_index=step_index,
                step_count=len(timeframes),
            )
        if not frames:
            return pd.DataFrame(columns=FORCE_DOWNLOAD_COLUMNS)
        return pd.concat(frames, ignore_index=True).loc[:, FORCE_DOWNLOAD_COLUMNS]


def normalize_symbol_tuple(symbols: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    return tuple(unique_symbols(tuple(symbols or ())))


def normalize_timeframes(timeframes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for item in timeframes:
        timeframe = ensure_supported_timeframe(item)
        if timeframe in seen:
            continue
        seen.add(timeframe)
        result.append(timeframe)
    if not result:
        raise ValueError("timeframes 不能为空。")
    return tuple(result)


def normalize_download_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in DOWNLOAD_MODES:
        raise ValueError("下载模式只支持 smart 或 force。")
    return normalized


def shortcut_symbols(name: str) -> tuple[str, ...]:
    return QUICK_SYMBOL_GROUPS.get(str(name), ())


def cache_summary(catalog: pd.DataFrame) -> dict[str, object]:
    summary = summarize_data_inventory(catalog)
    symbols = catalog["stock_code"].dropna().astype(str).nunique() if "stock_code" in catalog.columns else 0
    timeframes = catalog["timeframe"].dropna().astype(str).nunique() if "timeframe" in catalog.columns else 0
    asset_types = catalog["asset_type"].dropna().astype(str).nunique() if "asset_type" in catalog.columns else 0
    datasets = (
        catalog.loc[:, ["data_kind", "indicator"]].drop_duplicates().shape[0]
        if {"data_kind", "indicator"}.issubset(catalog.columns)
        else 0
    )
    latest_modified = catalog["modified_at"].max() if "modified_at" in catalog.columns and not catalog.empty else pd.NaT
    summary.update(
        {
            "symbol_count": float(symbols),
            "timeframe_count": float(timeframes),
            "asset_type_count": float(asset_types),
            "dataset_count": float(datasets),
            "catalog_row_count": float(len(catalog)),
            "latest_modified_at": latest_modified,
        }
    )
    return summary


def cache_by_timeframe(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["timeframe", "cached_count", "unavailable_count", "rows", "file_size_bytes", "latest_modified_at"]
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    frame = catalog.copy()
    frame["status"] = frame.get("status", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["rows"] = pd.to_numeric(frame.get("rows", pd.Series([0] * len(frame))), errors="coerce").fillna(0)
    frame["file_size_bytes"] = pd.to_numeric(
        frame.get("file_size_bytes", pd.Series([0] * len(frame))),
        errors="coerce",
    ).fillna(0)
    grouped = frame.groupby("timeframe", sort=False).agg(
        cached_count=("status", lambda values: int(values.eq("cached").sum())),
        row_count=("status", "size"),
        rows=("rows", "sum"),
        file_size_bytes=("file_size_bytes", "sum"),
        latest_modified_at=("modified_at", "max"),
    )
    grouped["unavailable_count"] = grouped["row_count"] - grouped["cached_count"]
    return grouped.reset_index().loc[:, columns]


def cache_by_status(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["status", "count", "rows", "file_size_bytes"]
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    frame = catalog.copy()
    frame["status"] = frame.get("status", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["rows"] = pd.to_numeric(frame.get("rows", pd.Series([0] * len(frame))), errors="coerce").fillna(0)
    frame["file_size_bytes"] = pd.to_numeric(
        frame.get("file_size_bytes", pd.Series([0] * len(frame))),
        errors="coerce",
    ).fillna(0)
    return (
        frame.groupby("status", sort=False)
        .agg(count=("status", "size"), rows=("rows", "sum"), file_size_bytes=("file_size_bytes", "sum"))
        .reset_index()
        .loc[:, columns]
    )


def cache_by_asset_type(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["asset_type", "asset_type_label", "cached_count", "unavailable_count", "rows", "file_size_bytes"]
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    frame = catalog.copy()
    frame["status"] = frame.get("status", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["rows"] = pd.to_numeric(frame.get("rows", pd.Series([0] * len(frame))), errors="coerce").fillna(0)
    frame["file_size_bytes"] = pd.to_numeric(
        frame.get("file_size_bytes", pd.Series([0] * len(frame))),
        errors="coerce",
    ).fillna(0)
    grouped = frame.groupby("asset_type", sort=False).agg(
        cached_count=("status", lambda values: int(values.eq("cached").sum())),
        row_count=("status", "size"),
        rows=("rows", "sum"),
        file_size_bytes=("file_size_bytes", "sum"),
    )
    grouped["unavailable_count"] = grouped["row_count"] - grouped["cached_count"]
    result = grouped.reset_index()
    result["asset_type_label"] = result["asset_type"].map(lambda value: ASSET_TYPE_LABELS.get(str(value), str(value)))
    return result.loc[:, columns]


def cache_by_dataset(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["asset_type", "data_kind", "indicator", "timeframe", "status", "count", "rows", "file_size_bytes"]
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    frame = catalog.copy()
    frame["rows"] = pd.to_numeric(frame.get("rows", pd.Series([0] * len(frame))), errors="coerce").fillna(0)
    frame["file_size_bytes"] = pd.to_numeric(
        frame.get("file_size_bytes", pd.Series([0] * len(frame))),
        errors="coerce",
    ).fillna(0)
    return (
        frame.groupby(["asset_type", "data_kind", "indicator", "timeframe", "status"], sort=False)
        .agg(count=("status", "size"), rows=("rows", "sum"), file_size_bytes=("file_size_bytes", "sum"))
        .reset_index()
        .loc[:, columns]
    )


def cache_readiness(catalog: pd.DataFrame) -> pd.DataFrame:
    """按资产类型和周期汇总回测准备度，帮助用户先看能否跑，再看明细。"""
    columns = [
        "timeframe",
        "asset_type",
        "asset_type_label",
        "total_count",
        "cached_count",
        "missing_count",
        "coverage_ratio",
        "earliest_start_at",
        "latest_end_at",
        "status",
        "message",
    ]
    if catalog.empty:
        return pd.DataFrame(columns=columns)

    frame = catalog.copy()
    frame["status"] = frame.get("status", pd.Series([""] * len(frame))).fillna("").astype(str)
    frame["start_at"] = pd.to_datetime(frame.get("start_at", pd.Series([pd.NaT] * len(frame))), errors="coerce")
    frame["end_at"] = pd.to_datetime(frame.get("end_at", pd.Series([pd.NaT] * len(frame))), errors="coerce")
    grouped = frame.groupby(["timeframe", "asset_type"], sort=False).agg(
        total_count=("status", "size"),
        cached_count=("status", lambda values: int(values.eq("cached").sum())),
        earliest_start_at=("start_at", "min"),
        latest_end_at=("end_at", "max"),
    )
    grouped["missing_count"] = grouped["total_count"] - grouped["cached_count"]
    grouped["coverage_ratio"] = grouped["cached_count"] / grouped["total_count"].where(grouped["total_count"].ne(0), 1)
    result = grouped.reset_index()
    result["asset_type_label"] = result["asset_type"].map(lambda value: ASSET_TYPE_LABELS.get(str(value), str(value)))
    result["status"] = [
        _cache_readiness_status(cached_count, total_count)
        for cached_count, total_count in zip(result["cached_count"], result["total_count"], strict=False)
    ]
    result["message"] = [
        _cache_readiness_message(status, missing_count)
        for status, missing_count in zip(result["status"], result["missing_count"], strict=False)
    ]
    return result.loc[:, columns]


def download_summary(table: pd.DataFrame) -> dict[str, object]:
    if table.empty:
        return {"row_count": 0.0, "fetched_count": 0.0, "cached_count": 0.0, "new_rows": 0.0, "rows_written": 0.0}
    action = table["action"].fillna("").astype(str) if "action" in table.columns else pd.Series([""] * len(table))
    new_rows = pd.to_numeric(table.get("new_rows", pd.Series([0] * len(table))), errors="coerce").fillna(0)
    rows_written = pd.to_numeric(table.get("rows_written", pd.Series([0] * len(table))), errors="coerce").fillna(0)
    return {
        "row_count": float(len(table)),
        "fetched_count": float(action.eq("fetched").sum()),
        "cached_count": float(action.eq("cached").sum()),
        "new_rows": float(new_rows.sum()),
        "rows_written": float(rows_written.sum()),
    }


def _cache_readiness_status(cached_count: object, total_count: object) -> str:
    cached = int(cached_count)
    total = int(total_count)
    if total <= 0 or cached <= 0:
        return "empty"
    if cached == total:
        return "ready"
    return "partial"


def _cache_readiness_message(status: str, missing_count: object) -> str:
    missing = int(missing_count)
    if status == "ready":
        return "缓存完整，可以进入回测前数据质量审计。"
    if status == "partial":
        return f"部分缓存可用，仍有 {missing} 项需要补齐。"
    return "没有可用缓存，回测前需要先补齐。"


def _force_download_frame(written: pd.DataFrame, *, timeframe: str, adjust: str) -> pd.DataFrame:
    if written.empty:
        return pd.DataFrame(columns=FORCE_DOWNLOAD_COLUMNS)
    frame = written.rename(columns={"symbol": "stock_code", "rows": "rows_written"}).copy()
    frame["timeframe"] = timeframe
    frame["adjust"] = adjust
    frame["action"] = "fetched"
    return frame.reindex(columns=FORCE_DOWNLOAD_COLUMNS)


def _emit_progress(callback: ProgressCallback | None, **payload: object) -> None:
    if callback is not None:
        callback(payload)
