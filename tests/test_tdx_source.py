from __future__ import annotations

import inspect

import pandas as pd
import pytest

from trending_winning.data import tdx as tdx_module
from trending_winning.data.tdx import diagnose_tdx_source, fetch_tdx_bars


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


class ErrorTq(FakeTq):
    def __init__(self) -> None:
        super().__init__({})

    def get_market_data(self, **kwargs: object) -> dict[str, object]:
        self.market_calls.append(kwargs)
        return {"error": -5, "msg": "period unsupported"}


class InitErrorTq(FakeTq):
    def __init__(self) -> None:
        super().__init__({})

    def initialize(self, caller_path: str) -> None:
        self.initialize_calls.append(caller_path)
        raise RuntimeError("not logged in")


class PeriodPayloadTq(FakeTq):
    def __init__(self, payloads_by_period: dict[str, dict[str, pd.DataFrame]]) -> None:
        super().__init__({})
        self.payloads_by_period = payloads_by_period

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.market_calls.append(kwargs)
        return self.payloads_by_period[str(kwargs["period"])]


def _payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-25 10:30:00", "2026-05-25 11:30:00"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [10.0, 10.6], "600519.SH": [100.0, 101.0]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.8, 11.4], "600519.SH": [102.0, 103.0]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.9, 10.4], "600519.SH": [99.0, 100.5]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [10.7, 11.2], "600519.SH": [101.5, 102.5]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000.0, 1200.0], "600519.SH": [500.0, 520.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [10700.0, 13440.0], "600519.SH": [50750.0, 53300.0]}, index=index),
    }


def _daily_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-24", "2026-05-25"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [9.8, 10.0]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [10.1, 10.8]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.6, 9.9]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [9.9, 10.7]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [9000.0, 10000.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [89100.0, 107000.0]}, index=index),
    }


def _empty_payload() -> dict[str, pd.DataFrame]:
    empty = pd.DataFrame({"000001.SZ": []}, index=pd.DatetimeIndex([]))
    return {field: empty.copy() for field in ("Open", "High", "Low", "Close", "Volume", "Amount")}


def _five_min_payload(symbol: str = "000001.SZ") -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-05-25 09:35:00", periods=12, freq="5min")
    opens = [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1]
    closes = [price + 0.05 for price in opens]
    highs = [price + 0.2 for price in opens]
    lows = [price - 0.1 for price in opens]
    volume = [100.0 + index for index in range(12)]
    amount = [vol * close for vol, close in zip(volume, closes, strict=False)]
    return {
        "Open": pd.DataFrame({symbol: opens}, index=index),
        "High": pd.DataFrame({symbol: highs}, index=index),
        "Low": pd.DataFrame({symbol: lows}, index=index),
        "Close": pd.DataFrame({symbol: closes}, index=index),
        "Volume": pd.DataFrame({symbol: volume}, index=index),
        "Amount": pd.DataFrame({symbol: amount}, index=index),
    }


def _partial_direct_30m_payload() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-25 10:00:00"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [20.0], "000002.SZ": [float("nan")]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [21.0], "000002.SZ": [float("nan")]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [19.5], "000002.SZ": [float("nan")]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [20.5], "000002.SZ": [float("nan")]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000.0], "000002.SZ": [float("nan")]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [20500.0], "000002.SZ": [float("nan")]}, index=index),
    }


def _direct_payload_omits_second_symbol() -> dict[str, pd.DataFrame]:
    index = pd.to_datetime(["2026-05-25 10:00:00"])
    return {
        "Open": pd.DataFrame({"000001.SZ": [20.0]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [21.0]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [19.5]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [20.5]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000.0]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [20500.0]}, index=index),
    }


def test_fetch_tdx_bars_supports_60m_batch_payload() -> None:
    fake = FakeTq(_payload())

    out = fetch_tdx_bars(
        symbols=("000001.SZ", "600519.SH"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        timeframe="60m",
        adjust="qfq",
        tq_client=fake,
    )

    assert fake.initialize_calls
    assert fake.refresh_calls == []
    assert fake.market_calls[0]["period"] == "1h"
    assert fake.market_calls[0]["stock_list"] == ["000001.SZ", "600519.SH"]
    assert fake.market_calls[0]["start_time"] == "20260525093000"
    assert fake.market_calls[0]["end_time"] == "20260525150000"
    assert fake.market_calls[0]["dividend_type"] == "front"
    assert list(out.columns) == ["date", "stock_code", "open", "high", "low", "close", "volume", "amount"]
    assert out["stock_code"].tolist() == ["000001.SZ", "000001.SZ", "600519.SH", "600519.SH"]
    assert out.loc[out["stock_code"] == "000001.SZ", "close"].tolist() == [10.7, 11.2]


def test_fetch_tdx_bars_supports_1d_daily_payload() -> None:
    fake = FakeTq(_daily_payload())

    out = fetch_tdx_bars(
        symbols=("000001.SZ",),
        start="2026-05-24",
        end="2026-05-25",
        timeframe="1d",
        adjust="qfq",
        tq_client=fake,
    )

    assert fake.market_calls[0]["period"] == "1d"
    assert fake.refresh_calls == []
    assert out["date"].tolist() == [pd.Timestamp("2026-05-24"), pd.Timestamp("2026-05-25")]
    assert out["close"].tolist() == [9.9, 10.7]


def test_fetch_tdx_bars_derives_30m_from_tdx_5m_when_direct_period_has_no_data() -> None:
    fake = PeriodPayloadTq({"30m": _empty_payload(), "5m": _five_min_payload()})

    out = fetch_tdx_bars(
        symbols=("000001.SZ",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 10:30:00",
        timeframe="30m",
        adjust="qfq",
        tq_client=fake,
    )

    assert [call["period"] for call in fake.market_calls] == ["30m", "5m"]
    assert fake.refresh_calls == [(["000001.SZ"], "5m")]
    assert out["date"].tolist() == [pd.Timestamp("2026-05-25 10:00:00"), pd.Timestamp("2026-05-25 10:30:00")]
    assert out["open"].tolist() == [10.0, 10.6]
    assert out["high"].tolist() == pytest.approx([10.7, 11.3])
    assert out["low"].tolist() == pytest.approx([9.9, 10.5])
    assert out["close"].tolist() == pytest.approx([10.55, 11.15])
    assert out["volume"].tolist() == pytest.approx([615.0, 651.0])
    assert out["amount"].tolist() == pytest.approx([6336.25, 7097.65])


def test_fetch_tdx_bars_derives_only_missing_symbols_from_5m_fallback() -> None:
    fake = PeriodPayloadTq({"30m": _partial_direct_30m_payload(), "5m": _five_min_payload("000002.SZ")})

    out = fetch_tdx_bars(
        symbols=("000001.SZ", "000002.SZ"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 10:30:00",
        timeframe="30m",
        adjust="qfq",
        tq_client=fake,
    )

    assert [call["period"] for call in fake.market_calls] == ["30m", "5m"]
    assert fake.market_calls[1]["stock_list"] == ["000002.SZ"]
    by_symbol = out.groupby("stock_code")["close"].apply(list).to_dict()
    assert by_symbol["000001.SZ"] == [20.5]
    assert by_symbol["000002.SZ"] == pytest.approx([10.55, 11.15])


def test_fetch_tdx_bars_derives_symbol_omitted_by_direct_payload_from_5m_fallback() -> None:
    fake = PeriodPayloadTq({"30m": _direct_payload_omits_second_symbol(), "5m": _five_min_payload("000002.SZ")})

    out = fetch_tdx_bars(
        symbols=("000001.SZ", "000002.SZ"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 10:30:00",
        timeframe="30m",
        adjust="qfq",
        tq_client=fake,
    )

    assert [call["period"] for call in fake.market_calls] == ["30m", "5m"]
    assert fake.market_calls[1]["stock_list"] == ["000002.SZ"]
    by_symbol = out.groupby("stock_code")["close"].apply(list).to_dict()
    assert by_symbol["000001.SZ"] == [20.5]
    assert by_symbol["000002.SZ"] == pytest.approx([10.55, 11.15])


def test_aggregate_5m_bars_uses_vectorized_bucket_assignment() -> None:
    source = inspect.getsource(tdx_module._aggregate_5m_bars)

    assert ".map(lambda" not in source
    assert ".apply(" not in source


def test_fetch_tdx_bars_initializes_new_client_even_when_python_reuses_id(monkeypatch) -> None:
    fake = FakeTq(_payload())
    monkeypatch.setattr(tdx_module, "_INITIALIZED_CLIENT_ID", id(fake))

    fetch_tdx_bars(
        symbols=("000001.SZ",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        timeframe="60m",
        adjust="qfq",
        tq_client=fake,
    )

    assert fake.initialize_calls


def test_fetch_tdx_bars_refreshes_5m_cache() -> None:
    fake = FakeTq(_payload())

    fetch_tdx_bars(
        symbols=("000001.SZ",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        timeframe="5m",
        adjust="qfq",
        tq_client=fake,
    )

    assert fake.refresh_calls == [(["000001.SZ"], "5m")]


def test_fetch_tdx_bars_rejects_non_target_timeframe() -> None:
    with pytest.raises(ValueError, match="timeframe"):
        fetch_tdx_bars(
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
            timeframe="1m",
            tq_client=FakeTq(_payload()),
        )


def test_fetch_tdx_bars_rejects_start_after_end_before_tdx_request() -> None:
    fake = FakeTq(_payload())

    with pytest.raises(ValueError, match="start 不能晚于 end"):
        fetch_tdx_bars(
            symbols=("000001.SZ",),
            start="2026-05-26",
            end="2026-05-25",
            timeframe="60m",
            tq_client=fake,
        )

    assert fake.initialize_calls == []
    assert fake.refresh_calls == []
    assert fake.market_calls == []


def test_fetch_tdx_bars_reports_tdx_error_payload() -> None:
    with pytest.raises(ValueError, match="period unsupported"):
        fetch_tdx_bars(
            symbols=("000001.SZ",),
            start="2026-05-25",
            end="2026-05-25",
            timeframe="60m",
            tq_client=ErrorTq(),
        )


def test_load_tq_rejects_macos_runtime_without_explicit_override(monkeypatch) -> None:
    monkeypatch.setattr(tdx_module.sys, "platform", "darwin")
    monkeypatch.delenv("TDX_ALLOW_MAC_TQCENTER", raising=False)
    monkeypatch.setattr(tdx_module, "_TQ_CLIENT", None)

    with pytest.raises(RuntimeError, match="Mac 通达信不支持"):
        tdx_module._load_tq("")


def test_diagnose_tdx_source_reports_each_timeframe_sample_request() -> None:
    fake = PeriodPayloadTq({"1d": _daily_payload(), "30m": _payload(), "1h": _payload()})

    report = diagnose_tdx_source(
        symbols=("000001.SZ",),
        timeframes=("1d", "30m", "60m"),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        tq_client=fake,
    )

    by_timeframe = report.set_index("timeframe")
    assert by_timeframe.loc["1d", "status"] == "ok"
    assert by_timeframe.loc["30m", "status"] == "ok"
    assert by_timeframe.loc["60m", "status"] == "ok"
    assert by_timeframe.loc["60m", "tdx_period"] == "1h"
    assert by_timeframe.loc["30m", "rows"] == 2
    assert by_timeframe.loc["30m", "start"] == pd.Timestamp("2026-05-25 10:30:00")
    assert by_timeframe.loc["30m", "end"] == pd.Timestamp("2026-05-25 11:30:00")
    assert [call["period"] for call in fake.market_calls] == ["1d", "30m", "1h"]


def test_diagnose_tdx_source_explains_intraday_no_data_as_windows_cache_issue() -> None:
    fake = PeriodPayloadTq({"5m": {}})

    report = diagnose_tdx_source(
        symbols=("000001.SZ",),
        timeframes=("5m",),
        start="2026-05-25 09:30:00",
        end="2026-05-25 15:00:00",
        tq_client=fake,
    )

    row = report.iloc[0]
    assert row["status"] == "no_data"
    assert "Parallels/Windows 通达信本地没有返回分钟 K 线" in row["message"]
    assert "Mac 本机通达信不参与取数" in row["message"]


def test_diagnose_tdx_source_rejects_start_after_end_before_initialization() -> None:
    fake = PeriodPayloadTq({"30m": _payload()})

    with pytest.raises(ValueError, match="start 不能晚于 end"):
        diagnose_tdx_source(
            symbols=("000001.SZ",),
            timeframes=("30m",),
            start="2026-05-26",
            end="2026-05-25",
            tq_client=fake,
        )

    assert fake.initialize_calls == []
    assert fake.market_calls == []


def test_diagnose_tdx_source_reports_initialization_error_per_timeframe() -> None:
    fake = InitErrorTq()

    report = diagnose_tdx_source(
        symbols=("000001.SZ",),
        timeframes=("30m", "60m"),
        start="2026-05-25",
        end="2026-05-25",
        tq_client=fake,
    )

    assert report["timeframe"].tolist() == ["30m", "60m"]
    assert report["status"].tolist() == ["init_error", "init_error"]
    assert report["rows"].tolist() == [0, 0]
    assert report["message"].str.contains("TDX 初始化失败").all()
