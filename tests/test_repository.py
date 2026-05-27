from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trending_winning.data.repository import (
    BacktestDataBundle,
    MultiTimeframeBacktestDataBundle,
    MarketDataRepository,
    audit_local_data,
    load_daily_bars,
    load_backtest_data,
    load_local_bars,
    load_multi_timeframe_backtest_data,
    plan_tdx_backtest_data,
    prepare_tdx_backtest_data,
    resolve_daily_root,
    resolve_timeframe_root,
    update_from_tdx,
    write_local_bars,
)
from trending_winning.data.schema import normalize_bars


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


class PeriodPayloadFakeTq(FakeTq):
    def __init__(self, payloads_by_period: dict[str, dict[str, pd.DataFrame]]) -> None:
        super().__init__({})
        self.payloads_by_period = payloads_by_period

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.market_calls.append(kwargs)
        return self.payloads_by_period[str(kwargs["period"])]


def _tdx_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-25 10:30:00", "2026-05-25 11:30:00"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [10.0, 10.6]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.8, 11.4]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.9, 10.4]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [10.7, 11.2]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000.0, 1200.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [10700.0, 13440.0]}, index=index),
    }


def _daily_tdx_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-24", "2026-05-25"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [9.8, 10.0]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.1, 10.8]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.6, 9.9]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [9.9, 10.7]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [9000.0, 10000.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [89100.0, 107000.0]}, index=index),
    }


def _two_session_daily_tdx_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-25", "2026-05-26"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [10.0, 10.2]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.8, 10.9]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.9, 10.0]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [10.7, 10.6]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [10000.0, 11000.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [107000.0, 116600.0]}, index=index),
    }


def _full_30m_tdx_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(
        [
            "2026-05-25 10:00:00",
            "2026-05-25 10:30:00",
            "2026-05-25 11:00:00",
            "2026-05-25 11:30:00",
            "2026-05-25 13:30:00",
            "2026-05-25 14:00:00",
            "2026-05-25 14:30:00",
            "2026-05-25 15:00:00",
            "2026-05-26 10:00:00",
            "2026-05-26 10:30:00",
            "2026-05-26 11:00:00",
            "2026-05-26 11:30:00",
            "2026-05-26 13:30:00",
            "2026-05-26 14:00:00",
            "2026-05-26 14:30:00",
            "2026-05-26 15:00:00",
        ]
    )
    return {
        "Open": pd.DataFrame({"000001.SZ": [10.0 + index * 0.1 for index in range(16)]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.3 + index * 0.1 for index in range(16)]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.9 + index * 0.1 for index in range(16)]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [10.2 + index * 0.1 for index in range(16)]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000.0 + index for index in range(16)]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [10200.0 + index * 100 for index in range(16)]}, index=index),
    }


def test_resolve_timeframe_root_maps_daily_root_to_minute_sibling(tmp_path: Path) -> None:
    root = tmp_path / "market" / "daily"

    assert resolve_timeframe_root(root, "5m") == tmp_path / "market" / "5m"
    assert resolve_timeframe_root(root, "60m") == tmp_path / "market" / "60m"
    assert resolve_timeframe_root(root, "1d") == tmp_path / "market" / "daily"
    assert resolve_timeframe_root(tmp_path / "market" / "30m", "1d") == tmp_path / "market" / "daily"
    assert resolve_timeframe_root(tmp_path / "market" / "30m", "60m") == tmp_path / "market" / "60m"


def test_write_local_bars_merges_and_deduplicates_by_symbol_date(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:30:00", "2026-05-25 11:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.5],
            "high": [10.8, 11.0],
            "low": [9.9, 10.4],
            "close": [10.7, 10.8],
            "volume": [1000.0, 1100.0],
            "amount": [10700.0, 11880.0],
        }
    )
    second = first.copy()
    second.loc[1, "close"] = 11.2

    first_result = write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=first)
    second_result = write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=second)
    loaded = load_local_bars(
        data_root=data_root,
        timeframe="60m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25 15:00:00",
    )

    assert first_result.loc[0, "rows"] == 2
    assert second_result.loc[0, "rows"] == 2
    assert loaded["close"].tolist() == [10.7, 11.2]
    assert (tmp_path / "market" / "60m" / "qfq" / "000001.SZ.parquet").exists()


def test_normalize_bars_drops_rows_with_missing_or_invalid_symbol() -> None:
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00", "2026-05-25 11:00:00"]),
            "stock_code": [None, "", "not-a-symbol"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10100.0, 11220.0, 12360.0],
        }
    )

    normalized = normalize_bars(bars)

    assert normalized.empty


def test_repository_data_loaders_reject_start_after_end(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start 不能晚于 end"):
        load_local_bars(
            data_root=tmp_path / "market" / "daily",
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-26",
            end="2026-05-25",
        )

    with pytest.raises(ValueError, match="start 不能晚于 end"):
        audit_local_data(
            data_root=tmp_path / "market" / "daily",
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-26",
            end="2026-05-25",
        )

    with pytest.raises(ValueError, match="start 不能晚于 end"):
        load_backtest_data(
            data_root=tmp_path / "market" / "daily",
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-26",
            end="2026-05-25",
        )


def test_daily_repository_uses_existing_market_daily_layout(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True)
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-24", "2026-05-25"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.6],
            "low": [9.9, 10.0],
            "close": [10.1, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10100.0, 11440.0],
        }
    )
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)
    repo = MarketDataRepository(data_root, adjust="qfq")

    loaded = load_daily_bars(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    assert resolve_daily_root(tmp_path / "market" / "60m") == tmp_path / "market" / "daily"
    assert loaded["close"].tolist() == [10.4]
    assert repo.load_daily_bars(symbols=("000001.SZ",), start="2026-05-24", end="2026-05-25")["close"].tolist() == [
        10.1,
        10.4,
    ]


def test_load_backtest_data_filters_daily_limit_open_sessions(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-24", "2026-05-25", "2026-05-26"]),
            "stock_code": ["300750.SZ", "300750.SZ", "300750.SZ"],
            "open": [10.0, 12.0, 12.2],
            "high": [10.1, 12.4, 12.8],
            "low": [9.8, 11.8, 12.0],
            "close": [10.0, 12.1, 12.5],
            "volume": [1000.0, 2000.0, 1800.0],
            "amount": [10000.0, 24200.0, 22500.0],
        }
    )
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-26 10:00:00"]),
            "stock_code": ["300750.SZ", "300750.SZ"],
            "open": [12.1, 12.3],
            "high": [12.2, 12.6],
            "low": [12.0, 12.1],
            "close": [12.15, 12.5],
            "volume": [900.0, 1000.0],
            "amount": [10935.0, 12500.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=intraday)
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_root / "300750.SZ.parquet", index=False)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("300750.SZ",),
        start="2026-05-25",
        end="2026-05-26",
    )

    assert isinstance(bundle, BacktestDataBundle)
    assert bundle.bars["date"].dt.normalize().unique().tolist() == [pd.Timestamp("2026-05-26")]
    assert bundle.filtered_limit_open_days["session_date"].tolist() == [pd.Timestamp("2026-05-25")]
    audit = bundle.limit_filter_audit.set_index("stock_code")
    assert audit.loc["300750.SZ", "status"] == "ok"
    assert audit.loc["300750.SZ", "daily_rows"] == 2
    assert audit.loc["300750.SZ", "filtered_days"] == 1


def test_load_backtest_data_reports_limit_filter_skipped_when_daily_bars_are_missing(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.1],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=intraday)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        strict_data_quality=False,
    )

    audit = bundle.limit_filter_audit.set_index("stock_code")
    assert audit.loc["000001.SZ", "status"] == "daily_missing"
    assert audit.loc["000001.SZ", "daily_rows"] == 0
    assert audit.loc["000001.SZ", "filtered_days"] == 0
    assert "无法判断一字涨停开盘过滤" in audit.loc["000001.SZ", "message"]


def test_load_backtest_data_fails_in_strict_mode_when_daily_filter_cannot_run(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.1],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=intraday)

    with pytest.raises(ValueError, match="日K一字涨停过滤未通过严格门禁.*000001\\.SZ=daily_missing"):
        load_backtest_data(
            data_root=data_root,
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
        )


def test_load_backtest_data_reports_limit_open_days_only_inside_requested_window(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-22", "2026-05-23", "2026-05-24", "2026-05-25"]),
            "stock_code": ["000001.SZ"] * 4,
            "open": [10.0, 11.0, 12.1, 12.3],
            "high": [10.1, 11.2, 12.3, 12.5],
            "low": [9.8, 10.9, 12.0, 12.1],
            "close": [10.0, 11.0, 12.0, 12.4],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "amount": [10000.0, 12100.0, 14400.0, 16120.0],
        }
    )
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ"],
            "open": [12.3],
            "high": [12.5],
            "low": [12.1],
            "close": [12.4],
            "volume": [1000.0],
            "amount": [12400.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=intraday)
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        strict_data_quality=False,
    )

    assert bundle.filtered_limit_open_days.empty
    assert bundle.limit_filter_audit.loc[0, "filtered_days"] == 0


def test_repository_updates_local_cache_from_tdx_source(tmp_path: Path) -> None:
    repo = MarketDataRepository(tmp_path / "market" / "daily", adjust="qfq")
    fake = PeriodPayloadFakeTq({"1d": _daily_tdx_payload(), "1h": _tdx_payload()})

    result = repo.update_from_tdx(
        symbols=("000001.SZ",),
        timeframe="60m",
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        tq_client=fake,
    )
    loaded = repo.load_bars(
        timeframe="60m",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    assert result.loc[0, "new_rows"] == 2
    assert fake.market_calls[0]["period"] == "1h"
    assert loaded["close"].tolist() == [10.7, 11.2]


def test_repository_updates_daily_cache_from_tdx_source(tmp_path: Path) -> None:
    repo = MarketDataRepository(tmp_path / "market" / "daily", adjust="qfq")
    fake = FakeTq(_daily_tdx_payload())

    result = repo.update_from_tdx(
        symbols=("000001.SZ",),
        timeframe="1d",
        start="2026-05-24",
        end="2026-05-25",
        tq_client=fake,
    )
    loaded = repo.load_daily_bars(symbols=("000001.SZ",), start="2026-05-24", end="2026-05-25")

    assert result.loc[0, "new_rows"] == 2
    assert fake.market_calls[0]["period"] == "1d"
    assert loaded["close"].tolist() == [9.9, 10.7]


def test_update_from_tdx_rejects_start_after_end_before_tdx_request(tmp_path: Path) -> None:
    fake = FakeTq(_tdx_payload())

    with pytest.raises(ValueError, match="start 不能晚于 end"):
        update_from_tdx(
            data_root=tmp_path / "market" / "daily",
            adjust="qfq",
            symbols=("000001.SZ",),
            timeframe="60m",
            start="2026-05-26",
            end="2026-05-25",
            tq_client=fake,
        )

    assert fake.refresh_calls == []
    assert fake.market_calls == []


def test_prepare_tdx_backtest_data_updates_only_failed_timeframes_and_reaudits(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    cached_30m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.3],
            "high": [10.4, 10.7],
            "low": [9.9, 10.2],
            "close": [10.2, 10.6],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11660.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=cached_30m)
    fake = PeriodPayloadFakeTq({"1d": _daily_tdx_payload(), "1h": _tdx_payload()})

    summary = prepare_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("30m", "60m"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        tq_client=fake,
    )

    by_key = summary.set_index(["timeframe", "stock_code"])
    assert by_key.loc[("1d", "000001.SZ"), "action"] == "fetched"
    assert by_key.loc[("30m", "000001.SZ"), "action"] == "cached"
    assert by_key.loc[("30m", "000001.SZ"), "before_status"] == "ok"
    assert by_key.loc[("60m", "000001.SZ"), "action"] == "fetched"
    assert by_key.loc[("60m", "000001.SZ"), "before_status"] == "missing_file"
    assert by_key.loc[("60m", "000001.SZ"), "after_status"] == "ok"
    assert by_key.loc[("60m", "000001.SZ"), "new_rows"] == 2
    assert [call["period"] for call in fake.market_calls] == ["1d", "1h"]
    assert load_local_bars(
        data_root=data_root,
        timeframe="60m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )["close"].tolist() == [10.7, 11.2]


def test_prepare_tdx_backtest_data_can_fetch_missing_daily_parquet(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    fake = FakeTq(_daily_tdx_payload())

    summary = prepare_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("1d",),
        start="2026-05-24",
        end="2026-05-25",
        tq_client=fake,
    )

    row = summary.iloc[0]
    assert row["timeframe"] == "1d"
    assert row["action"] == "fetched"
    assert row["before_status"] == "missing_file"
    assert row["after_status"] == "ok"
    assert row["new_rows"] == 2
    assert fake.market_calls[0]["period"] == "1d"
    loaded = load_daily_bars(data_root=data_root, adjust="qfq", symbols=("000001.SZ",), start="2026-05-24", end="2026-05-25")
    assert loaded["close"].tolist() == [9.9, 10.7]


def test_prepare_tdx_backtest_data_fetches_daily_dependency_for_intraday_request(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    fake = PeriodPayloadFakeTq({"1d": _daily_tdx_payload(), "30m": _full_30m_tdx_payload()})

    summary = prepare_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("30m",),
        start="2026-05-25",
        end="2026-05-25 15:00:00",
        tq_client=fake,
    )

    by_key = summary.set_index(["timeframe", "stock_code"])
    assert by_key.loc[("1d", "000001.SZ"), "action"] == "fetched"
    assert by_key.loc[("30m", "000001.SZ"), "action"] == "fetched"
    assert [call["period"] for call in fake.market_calls] == ["1d", "30m"]


def test_prepare_tdx_backtest_data_refetches_coverage_below_minimum(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    sparse = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "high": [10.4],
            "low": [9.9],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=sparse)
    full_index = pd.to_datetime(
        [
            "2026-05-25 10:00:00",
            "2026-05-25 10:30:00",
            "2026-05-25 11:00:00",
            "2026-05-25 11:30:00",
            "2026-05-25 13:30:00",
            "2026-05-25 14:00:00",
            "2026-05-25 14:30:00",
            "2026-05-25 15:00:00",
        ]
    )
    fake = PeriodPayloadFakeTq(
        {
            "1d": _daily_tdx_payload(),
            "30m": {
                "Open": pd.DataFrame({"000001.SZ": [10.0 + index * 0.1 for index in range(8)]}, index=full_index),
                "High": pd.DataFrame({"000001.SZ": [10.3 + index * 0.1 for index in range(8)]}, index=full_index),
                "Low": pd.DataFrame({"000001.SZ": [9.9 + index * 0.1 for index in range(8)]}, index=full_index),
                "Close": pd.DataFrame({"000001.SZ": [10.2 + index * 0.1 for index in range(8)]}, index=full_index),
                "Volume": pd.DataFrame({"000001.SZ": [1000.0 + index for index in range(8)]}, index=full_index),
                "Amount": pd.DataFrame({"000001.SZ": [10200.0 + index * 100 for index in range(8)]}, index=full_index),
            },
        }
    )

    summary = MarketDataRepository(data_root, adjust="qfq").prepare_from_tdx(
        symbols=("000001.SZ",),
        timeframes=("30m",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        min_coverage_ratio=0.95,
        tq_client=fake,
    )

    row = summary.set_index(["timeframe", "stock_code"]).loc[("30m", "000001.SZ")]
    assert row["action"] == "fetched"
    assert row["before_status"] == "coverage_below_min"
    assert row["after_status"] == "ok"
    assert row["coverage_ratio"] == pytest.approx(1.0)
    assert row["new_rows"] == 7


def test_prepare_tdx_backtest_data_reports_before_and_after_coverage_delta(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    sparse = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "high": [10.4],
            "low": [9.9],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=sparse)
    full_index = pd.to_datetime(
        [
            "2026-05-25 10:00:00",
            "2026-05-25 10:30:00",
            "2026-05-25 11:00:00",
            "2026-05-25 11:30:00",
            "2026-05-25 13:30:00",
            "2026-05-25 14:00:00",
            "2026-05-25 14:30:00",
            "2026-05-25 15:00:00",
        ]
    )
    fake = PeriodPayloadFakeTq(
        {
            "1d": _daily_tdx_payload(),
            "30m": {
                "Open": pd.DataFrame({"000001.SZ": [10.0 + index * 0.1 for index in range(8)]}, index=full_index),
                "High": pd.DataFrame({"000001.SZ": [10.3 + index * 0.1 for index in range(8)]}, index=full_index),
                "Low": pd.DataFrame({"000001.SZ": [9.9 + index * 0.1 for index in range(8)]}, index=full_index),
                "Close": pd.DataFrame({"000001.SZ": [10.2 + index * 0.1 for index in range(8)]}, index=full_index),
                "Volume": pd.DataFrame({"000001.SZ": [1000.0 + index for index in range(8)]}, index=full_index),
                "Amount": pd.DataFrame({"000001.SZ": [10200.0 + index * 100 for index in range(8)]}, index=full_index),
            },
        }
    )

    summary = prepare_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("30m",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        min_coverage_ratio=0.95,
        tq_client=fake,
    )

    row = summary.set_index(["timeframe", "stock_code"]).loc[("30m", "000001.SZ")]
    assert row["before_missing_rows"] == 7
    assert row["after_missing_rows"] == 0
    assert row["before_coverage_ratio"] == pytest.approx(0.125)
    assert row["after_coverage_ratio"] == pytest.approx(1.0)
    assert row["before_max_missing_gap_minutes"] == 210
    assert row["after_max_missing_gap_minutes"] == 0


def test_prepare_tdx_backtest_data_fetches_daily_before_auditing_intraday_coverage(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    first_session = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0] * 8,
            "amount": [10200.0] * 8,
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=first_session)
    fake = PeriodPayloadFakeTq({"1d": _two_session_daily_tdx_payload(), "30m": _full_30m_tdx_payload()})

    summary = prepare_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("1d", "30m"),
        start="2026-05-25",
        end="2026-05-26 15:00:00",
        min_coverage_ratio=0.95,
        tq_client=fake,
    )

    by_key = summary.set_index(["timeframe", "stock_code"])
    assert by_key.loc[("1d", "000001.SZ"), "action"] == "fetched"
    assert by_key.loc[("30m", "000001.SZ"), "action"] == "fetched"
    assert by_key.loc[("30m", "000001.SZ"), "before_status"] == "coverage_below_min"
    assert by_key.loc[("30m", "000001.SZ"), "missing_rows"] == 0
    assert by_key.loc[("30m", "000001.SZ"), "coverage_ratio"] == pytest.approx(1.0)
    assert [call["period"] for call in fake.market_calls] == ["1d", "30m"]


def test_plan_tdx_backtest_data_reports_cached_and_fetch_actions_without_tdx_client(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    cached_30m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.1],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=cached_30m)

    plan = plan_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("5m", "15m", "30m", "60m"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
    )

    by_key = plan.set_index(["timeframe", "stock_code"])
    assert by_key.loc[("1d", "000001.SZ"), "action"] == "fetch"
    assert by_key.loc[("30m", "000001.SZ"), "action"] == "cached"
    assert by_key.loc[("30m", "000001.SZ"), "reason"] == "local_ok"
    assert by_key.loc[("5m", "000001.SZ"), "action"] == "fetch"
    assert by_key.loc[("15m", "000001.SZ"), "action"] == "fetch"
    assert by_key.loc[("60m", "000001.SZ"), "action"] == "fetch"
    assert by_key.loc[("60m", "000001.SZ"), "before_status"] == "missing_file"


def test_plan_tdx_backtest_data_uses_daily_sessions_to_find_whole_day_intraday_gaps(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True)
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.8],
            "low": [9.9, 10.0],
            "close": [10.2, 10.6],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12720.0],
        }
    )
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)
    first_session = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0] * 8,
            "amount": [10200.0] * 8,
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=first_session)

    plan = MarketDataRepository(data_root, adjust="qfq").plan_from_tdx(
        symbols=("000001.SZ",),
        timeframes=("30m",),
        start="2026-05-25",
        end="2026-05-26 15:00:00",
        min_coverage_ratio=0.95,
    )

    row = plan.set_index(["timeframe", "stock_code"]).loc[("30m", "000001.SZ")]
    assert row["action"] == "fetch"
    assert row["reason"] == "coverage_below_min"
    assert row["expected_rows"] == 16
    assert row["missing_rows"] == 8
    assert row["coverage_ratio"] == pytest.approx(0.5)


def test_repository_audit_bars_uses_daily_sessions_to_find_whole_day_intraday_gaps(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.8],
            "low": [9.9, 10.0],
            "close": [10.2, 10.6],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12720.0],
        }
    )
    first_session = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0] * 8,
            "amount": [10200.0] * 8,
        }
    )
    write_local_bars(data_root=data_root, timeframe="1d", adjust="qfq", bars=daily)
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=first_session)

    audit = MarketDataRepository(data_root, adjust="qfq").audit_bars(
        timeframe="30m",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-26 15:00:00",
    )

    row = audit.iloc[0]
    assert row["expected_rows"] == 16
    assert row["missing_rows"] == 8
    assert row["coverage_ratio"] == pytest.approx(0.5)


def test_plan_tdx_backtest_data_keeps_start_session_when_start_has_intraday_time(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True)
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.8],
            "low": [9.9, 10.0],
            "close": [10.2, 10.6],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12720.0],
        }
    )
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)
    first_session = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0] * 8,
            "amount": [10200.0] * 8,
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=first_session)

    plan = plan_tdx_backtest_data(
        data_root=data_root,
        adjust="qfq",
        symbols=("000001.SZ",),
        timeframes=("30m",),
        start="2026-05-25 09:30:00",
        end="2026-05-26 15:00:00",
        min_coverage_ratio=0.95,
    )

    row = plan.set_index(["timeframe", "stock_code"]).loc[("30m", "000001.SZ")]
    assert row["expected_rows"] == 16
    assert row["missing_rows"] == 8
    assert row["coverage_ratio"] == pytest.approx(0.5)


def test_audit_local_data_reports_coverage_and_quality_by_symbol(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    good = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.3],
            "high": [10.4, 10.6],
            "low": [9.9, 10.2],
            "close": [10.2, 10.5],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12600.0],
        }
    )
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000003.SZ", "000003.SZ"],
            "open": [0.0, 10.0],
            "high": [10.2, 10.2],
            "low": [9.8, 9.8],
            "close": [10.1, pd.NA],
            "volume": [1000.0, 1000.0],
            "amount": [10100.0, 10100.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=good)
    root = resolve_timeframe_root(data_root, "30m") / "qfq"
    bad.to_parquet(root / "000003.SZ.parquet", index=False)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ", "000002.SZ", "000003.SZ"),
        start="2026-05-25",
        end="2026-05-25",
    )

    by_symbol = audit.set_index("stock_code")
    assert by_symbol.loc["000001.SZ", "status"] == "ok"
    assert by_symbol.loc["000001.SZ", "rows_in_window"] == 2
    assert by_symbol.loc["000002.SZ", "status"] == "missing_file"
    assert by_symbol.loc["000003.SZ", "status"] == "quality_error"
    assert by_symbol.loc["000003.SZ", "duplicate_rows"] == 1
    assert by_symbol.loc["000003.SZ", "non_positive_price_rows"] == 1
    assert by_symbol.loc["000003.SZ", "null_ohlc_rows"] == 1


def test_audit_local_data_rejects_inconsistent_ohlc_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    inconsistent = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.3],
            "high": [9.9, 10.6],
            "low": [9.8, 10.4],
            "close": [10.2, 10.2],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12600.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=inconsistent)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "quality_error"
    assert row["inconsistent_ohlc_rows"] == 2
    assert "OHLC" in row["message"]


def test_audit_local_data_rejects_bad_volume_and_amount_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bad_turnover = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.3],
            "high": [10.4, 10.6],
            "low": [9.9, 10.2],
            "close": [10.2, 10.5],
            "volume": [-1.0, pd.NA],
            "amount": [10200.0, -100.0],
        }
    )
    root = resolve_timeframe_root(data_root, "30m") / "qfq"
    root.mkdir(parents=True)
    bad_turnover.to_parquet(root / "000001.SZ.parquet", index=False)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "quality_error"
    assert row["null_volume_amount_rows"] == 1
    assert row["negative_volume_amount_rows"] == 2


def test_audit_local_data_reports_all_zero_liquidity_rows_as_no_window_data(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    zero_turnover = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.3],
            "high": [10.4, 10.6],
            "low": [9.9, 10.2],
            "close": [10.2, 10.5],
            "volume": [0.0, 1200.0],
            "amount": [10200.0, 0.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=zero_turnover)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "no_window_data"
    assert row["zero_volume_amount_rows"] == 2
    assert row["null_volume_amount_rows"] == 0
    assert row["negative_volume_amount_rows"] == 0
    assert row["rows_in_window"] == 0


def test_audit_local_data_counts_zero_liquidity_rows_as_missing_coverage(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "amount": [10200.0, 0.0, 10200.0, 10200.0, 10200.0, 10200.0, 10200.0, 10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "ok"
    assert row["zero_volume_amount_rows"] == 1
    assert row["rows_in_window"] == 7
    assert row["expected_rows"] == 8
    assert row["missing_rows"] == 1
    assert row["coverage_ratio"] == pytest.approx(0.875)


def test_load_backtest_data_min_coverage_ratio_uses_tradable_coverage(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.4] * 8,
            "low": [9.8] * 8,
            "close": [10.2] * 8,
            "volume": [1000.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            "amount": [10200.0, 0.0, 10200.0, 10200.0, 10200.0, 10200.0, 10200.0, 10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    with pytest.raises(ValueError, match=r"000001\.SZ/30m=coverage_below_min\(0\.875 < 1\)"):
        load_backtest_data(
            data_root=data_root,
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
            min_coverage_ratio=1.0,
            filter_limit_open=False,
        )


def test_load_backtest_data_removes_zero_liquidity_rows_from_returned_bars(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00", "2026-05-25 11:00:00"]),
            "stock_code": ["000001.SZ"] * 3,
            "open": [10.0, 10.2, 10.4],
            "high": [10.3, 10.5, 10.7],
            "low": [9.9, 10.1, 10.3],
            "close": [10.2, 10.4, 10.6],
            "volume": [1000.0, 0.0, 1200.0],
            "amount": [10200.0, 0.0, 12720.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        filter_limit_open=False,
    )

    assert bundle.bars["date"].tolist() == [pd.Timestamp("2026-05-25 10:00:00"), pd.Timestamp("2026-05-25 11:00:00")]
    assert bundle.data_audit.loc[0, "zero_volume_amount_rows"] == 1


def test_load_multi_timeframe_backtest_data_removes_zero_liquidity_rows_per_timeframe(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars_30m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.9, 10.1],
            "close": [10.2, 10.4],
            "volume": [0.0, 1100.0],
            "amount": [0.0, 11440.0],
        }
    )
    bars_60m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:30:00", "2026-05-25 11:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.2, 10.5],
            "high": [10.6, 10.8],
            "low": [10.1, 10.4],
            "close": [10.5, 10.7],
            "volume": [1200.0, 0.0],
            "amount": [12600.0, 0.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars_30m)
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=bars_60m)

    bundle = load_multi_timeframe_backtest_data(
        data_root=data_root,
        timeframes=("30m", "60m"),
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        filter_limit_open=False,
    )

    assert bundle.bars_by_timeframe["30m"]["date"].tolist() == [pd.Timestamp("2026-05-25 10:30:00")]
    assert bundle.bars_by_timeframe["60m"]["date"].tolist() == [pd.Timestamp("2026-05-25 10:30:00")]
    by_timeframe = bundle.data_audit.set_index("timeframe")
    assert by_timeframe.loc["30m", "zero_volume_amount_rows"] == 1
    assert by_timeframe.loc["60m", "zero_volume_amount_rows"] == 1


def test_audit_local_data_rejects_invalid_date_or_symbol_rows(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    root = resolve_timeframe_root(data_root, "30m") / "qfq"
    root.mkdir(parents=True)
    malformed_identity = pd.DataFrame(
        {
            "date": [pd.NaT, pd.Timestamp("2026-05-25 10:30:00")],
            "stock_code": ["000001.SZ", ""],
            "open": [10.0, 10.3],
            "high": [10.4, 10.6],
            "low": [9.9, 10.2],
            "close": [10.2, 10.5],
            "volume": [1000.0, 1200.0],
            "amount": [10200.0, 12600.0],
        }
    )
    malformed_identity.to_parquet(root / "000001.SZ.parquet", index=False)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "quality_error"
    assert row["invalid_date_rows"] == 1
    assert row["invalid_symbol_rows"] == 1
    assert "日期或标的代码异常" in row["message"]


def test_audit_local_data_reports_intraday_session_coverage_gaps(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 7,
            "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "high": [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
            "low": [9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
            "volume": [1000.0] * 7,
            "amount": [10100.0, 10200.0, 10300.0, 10400.0, 10500.0, 10600.0, 10700.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    audit = audit_local_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
    )

    row = audit.iloc[0]
    assert row["status"] == "ok"
    assert row["rows_in_window"] == 7
    assert row["expected_rows"] == 8
    assert row["missing_rows"] == 1
    assert row["coverage_ratio"] == pytest.approx(0.875)
    assert row["max_missing_gap_minutes"] == 30


def test_load_backtest_data_fails_when_min_coverage_ratio_is_not_met(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 7,
            "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "high": [10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
            "low": [9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
            "volume": [1000.0] * 7,
            "amount": [10100.0, 10200.0, 10300.0, 10400.0, 10500.0, 10600.0, 10700.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    with pytest.raises(ValueError, match=r"000001\.SZ/30m=coverage_below_min\(0\.875 < 0\.95\)"):
        load_backtest_data(
            data_root=data_root,
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
            min_coverage_ratio=0.95,
            filter_limit_open=False,
        )

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        min_coverage_ratio=0.8,
        filter_limit_open=False,
    )
    assert bundle.data_audit.loc[0, "coverage_ratio"] == pytest.approx(0.875)


def test_load_backtest_data_uses_daily_sessions_to_detect_missing_intraday_day(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.6],
            "low": [9.9, 10.0],
            "close": [10.1, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10100.0, 11440.0],
        }
    )
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:30:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:30:00",
                    "2026-05-25 13:30:00",
                    "2026-05-25 14:00:00",
                    "2026-05-25 14:30:00",
                    "2026-05-25 15:00:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 8,
            "open": [10.0] * 8,
            "high": [10.2] * 8,
            "low": [9.9] * 8,
            "close": [10.1] * 8,
            "volume": [1000.0] * 8,
            "amount": [10100.0] * 8,
        }
    )
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=intraday)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-26",
        min_coverage_ratio=0.5,
    )

    row = bundle.data_audit.iloc[0]
    assert row["expected_rows"] == 16
    assert row["missing_rows"] == 8
    assert row["coverage_ratio"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match=r"000001\.SZ/30m=coverage_below_min\(0\.5 < 0\.75\)"):
        load_backtest_data(
            data_root=data_root,
            timeframe="30m",
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-26",
            min_coverage_ratio=0.75,
        )


def test_load_backtest_data_fails_fast_on_bad_local_data_by_default(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    root = resolve_timeframe_root(data_root, "30m") / "qfq"
    root.mkdir(parents=True)
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000003.SZ", "000003.SZ"],
            "open": [0.0, 10.0],
            "high": [10.2, 10.2],
            "low": [9.8, 9.8],
            "close": [10.1, pd.NA],
            "volume": [1000.0, 1000.0],
            "amount": [10100.0, 10100.0],
        }
    )
    bad.to_parquet(root / "000003.SZ.parquet", index=False)

    with pytest.raises(ValueError, match="000003.SZ.*quality_error"):
        load_backtest_data(
            data_root=data_root,
            timeframe="30m",
            adjust="qfq",
            symbols=("000003.SZ",),
            start="2026-05-25",
            end="2026-05-25",
        )


def test_load_backtest_data_can_explicitly_disable_quality_gate(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    root = resolve_timeframe_root(data_root, "30m") / "qfq"
    root.mkdir(parents=True)
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000003.SZ", "000003.SZ"],
            "open": [0.0, 10.0],
            "high": [10.2, 10.2],
            "low": [9.8, 9.8],
            "close": [10.1, pd.NA],
            "volume": [1000.0, 1000.0],
            "amount": [10100.0, 10100.0],
        }
    )
    bad.to_parquet(root / "000003.SZ.parquet", index=False)

    bundle = load_backtest_data(
        data_root=data_root,
        timeframe="30m",
        adjust="qfq",
        symbols=("000003.SZ",),
        start="2026-05-25",
        end="2026-05-25",
        strict_data_quality=False,
    )

    assert bundle.data_audit.loc[0, "status"] == "quality_error"
    assert len(bundle.bars) == 1


def test_load_multi_timeframe_backtest_data_audits_and_filters_each_timeframe(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-24", "2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 11.0, 11.1],
            "high": [10.3, 11.2, 11.4],
            "low": [9.8, 10.8, 10.9],
            "close": [10.0, 11.1, 11.2],
            "volume": [1000.0, 2000.0, 1800.0],
            "amount": [10000.0, 22200.0, 20160.0],
        }
    )
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_root / "000001.SZ.parquet", index=False)
    for timeframe, minutes in {"5m": 5, "30m": 30}.items():
        rows: list[dict[str, object]] = []
        for session in ("2026-05-25", "2026-05-26"):
            for index in range(2):
                close = 11.0 + index * 0.1
                rows.append(
                    {
                        "date": pd.Timestamp(f"{session} 09:30:00") + pd.Timedelta(minutes=minutes * index),
                        "stock_code": "000001.SZ",
                        "open": close - 0.05,
                        "high": close + 0.10,
                        "low": close - 0.10,
                        "close": close,
                        "volume": 1000.0,
                        "amount": close * 1000.0,
                    }
                )
        write_local_bars(data_root=data_root, timeframe=timeframe, adjust="qfq", bars=pd.DataFrame(rows))

    bundle = load_multi_timeframe_backtest_data(
        data_root=data_root,
        timeframes=("5m", "30m"),
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-26",
    )

    assert isinstance(bundle, MultiTimeframeBacktestDataBundle)
    assert set(bundle.bars_by_timeframe) == {"5m", "30m"}
    assert len(bundle.data_audit) == 2
    assert bundle.data_audit["status"].tolist() == ["ok", "ok"]
    assert bundle.filtered_limit_open_days["session_date"].tolist() == [pd.Timestamp("2026-05-25")]
    for bars in bundle.bars_by_timeframe.values():
        assert bars["date"].dt.normalize().unique().tolist() == [pd.Timestamp("2026-05-26")]


def test_load_multi_timeframe_backtest_data_fails_when_daily_filter_cannot_run(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    for timeframe in ("5m", "30m"):
        write_local_bars(
            data_root=data_root,
            timeframe=timeframe,
            adjust="qfq",
            bars=pd.DataFrame(
                {
                    "date": pd.to_datetime(["2026-05-25 10:00:00"]),
                    "stock_code": ["000001.SZ"],
                    "open": [10.0],
                    "high": [10.2],
                    "low": [9.9],
                    "close": [10.1],
                    "volume": [1000.0],
                    "amount": [10100.0],
                }
            ),
        )

    with pytest.raises(ValueError, match="日K一字涨停过滤未通过严格门禁.*000001\\.SZ=daily_missing"):
        load_multi_timeframe_backtest_data(
            data_root=data_root,
            timeframes=("5m", "30m"),
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
        )


def test_load_multi_timeframe_backtest_data_fails_if_any_timeframe_fails_audit(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    write_local_bars(
        data_root=data_root,
        timeframe="5m",
        adjust="qfq",
        bars=pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-05-25 09:30:00"]),
                "stock_code": ["000001.SZ"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000.0],
                "amount": [10100.0],
            }
        ),
    )

    with pytest.raises(ValueError, match="30m.*missing_file"):
        load_multi_timeframe_backtest_data(
            data_root=data_root,
            timeframes=("5m", "30m"),
            adjust="qfq",
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
        )
