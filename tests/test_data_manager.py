from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from trending_winning.data.catalog import query_catalog
from trending_winning.data.manager import DataDownloadConfig, DataManagementService, shortcut_symbols
from trending_winning.data.storage import write_local_bars


class FakeTq:
    def __init__(self, payload: dict[str, pd.DataFrame]) -> None:
        self.payload = payload
        self.initialize_calls: list[str] = []
        self.market_calls: list[dict[str, object]] = []
        self.refresh_calls: list[tuple[list[str], str]] = []

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)

    def refresh_kline(self, stock_list: list[str], period: str) -> str:
        self.refresh_calls.append((stock_list, period))
        return '{"ErrorId":"0","Msg":"ok"}'

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.market_calls.append(kwargs)
        return self.payload


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:30:00", "2026-05-25 11:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.6],
            "high": [10.8, 11.4],
            "low": [9.9, 10.4],
            "close": [10.7, 11.2],
            "volume": [1000.0, 1200.0],
            "amount": [10700.0, 13440.0],
        }
    )


def _tdx_payload() -> dict[str, pd.DataFrame]:
    bars = _bars().set_index("date")
    return {
        "Open": pd.DataFrame({"000001.SZ": bars["open"]}),
        "High": pd.DataFrame({"000001.SZ": bars["high"]}),
        "Low": pd.DataFrame({"000001.SZ": bars["low"]}),
        "Close": pd.DataFrame({"000001.SZ": bars["close"]}),
        "Volume": pd.DataFrame({"000001.SZ": bars["volume"]}),
        "Amount": pd.DataFrame({"000001.SZ": bars["amount"]}),
    }


def test_data_management_service_summarizes_cache_snapshot(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=_bars())
    metadata = data_root.parent / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "symbols.csv").write_text(
        "stock_code,stock_name\n000001.SZ,平安银行\n510300.SH,沪深300ETF\n000300.SH,沪深300\n",
        encoding="utf-8",
    )

    snapshot = DataManagementService(data_root, adjust="qfq").cache_snapshot(
        timeframes=("60m",),
        symbols=("000001.SZ", "510300.SH", "000300.SH"),
    )

    assert snapshot.summary["symbol_count"] == 3.0
    assert snapshot.summary["asset_type_count"] == 3.0
    assert snapshot.summary["dataset_count"] == 1.0
    assert snapshot.summary["data_inventory_cached_count"] == 1.0
    assert snapshot.summary["data_inventory_missing_file_count"] == 2.0
    assert snapshot.catalog_path.exists()
    with sqlite3.connect(snapshot.catalog_path) as connection:
        indexes = pd.read_sql_query("PRAGMA index_list(market_data_files)", connection)
    assert "idx_market_data_lookup" in indexes["name"].tolist()
    by_timeframe = snapshot.by_timeframe.set_index("timeframe")
    assert by_timeframe.loc["60m", "cached_count"] == 1
    assert by_timeframe.loc["60m", "unavailable_count"] == 2
    by_asset_type = snapshot.by_asset_type.set_index("asset_type")
    assert by_asset_type.loc["stock", "cached_count"] == 1
    assert by_asset_type.loc["etf", "unavailable_count"] == 1
    assert by_asset_type.loc["index", "unavailable_count"] == 1
    queried = query_catalog(data_root=data_root, asset_types=("stock",), timeframes=("60m",))
    assert queried["stock_code"].tolist() == ["000001.SZ"]


def test_data_management_service_force_download_uses_batch_and_progress(tmp_path: Path) -> None:
    service = DataManagementService(tmp_path / "market" / "daily", adjust="qfq")
    fake = FakeTq(_tdx_payload())
    events: list[dict[str, object]] = []

    result = service.repository.update_from_tdx(
        symbols=("000001.SZ",),
        timeframe="60m",
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        tq_client=fake,
        batch_size=1,
        progress_callback=events.append,
    )

    assert result.loc[0, "new_rows"] == 2
    assert fake.market_calls[0]["stock_list"] == ["000001.SZ"]
    assert "fetch_start" in [event["stage"] for event in events]
    assert "write_done" in [event["stage"] for event in events]


def test_data_management_service_download_rejects_unknown_mode(tmp_path: Path) -> None:
    service = DataManagementService(tmp_path / "market" / "daily", adjust="qfq")
    config = DataDownloadConfig(
        symbols=("000001.SZ",),
        timeframes=("60m",),
        start="2026-05-25",
        end="2026-05-25",
    )

    try:
        service.download(config, mode="other")
    except ValueError as exc:
        assert "下载模式只支持" in str(exc)
    else:
        raise AssertionError("unknown mode should fail")


def test_shortcut_symbols_exposes_non_manual_groups() -> None:
    assert "000300.SH" in shortcut_symbols("宽基指数")
    assert "510300.SH" in shortcut_symbols("ETF样例")
