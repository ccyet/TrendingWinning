from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.test_backtest import _bars
from trending_winning.data.repository import write_local_bars
from trending_winning.multitimeframe import scan_timeframes
from trending_winning.strategy import StrategyConfig


def test_scan_timeframes_returns_latest_state_per_symbol_and_timeframe(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.4, 10.5],
            "low": [9.8, 10.0],
            "close": [10.1, 10.3],
            "volume": [1000.0, 1200.0],
            "amount": [10100.0, 12360.0],
        }
    ).to_parquet(daily_root / "000001.SZ.parquet", index=False)
    bars_30m = _bars()
    bars_60m = _bars().assign(close=lambda frame: frame["close"] * 1.01)
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars_30m)
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=bars_60m)

    result = scan_timeframes(
        data_root=data_root,
        timeframes=("30m", "60m"),
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-26",
        strategy=StrategyConfig(channel_lookback=8, landmark_lookback=6),
    )

    assert set(result.full["timeframe"]) == {"30m", "60m"}
    assert result.latest[["timeframe", "stock_code"]].to_dict("records") == [
        {"timeframe": "30m", "stock_code": "000001.SZ"},
        {"timeframe": "60m", "stock_code": "000001.SZ"},
    ]
    assert {"channel_direction", "breakout_trigger", "channel_upper"}.issubset(result.latest.columns)


def test_scan_timeframes_uses_data_audit_and_limit_open_filter(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    daily_root = data_root / "qfq"
    daily_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-24", "2026-05-25", "2026-05-26"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 11.0, 11.1],
            "high": [10.2, 11.2, 11.4],
            "low": [9.8, 10.8, 10.9],
            "close": [10.0, 11.1, 11.3],
            "volume": [1000.0, 2000.0, 1800.0],
            "amount": [10000.0, 22200.0, 20340.0],
        }
    ).to_parquet(daily_root / "000001.SZ.parquet", index=False)
    for timeframe, minutes in {"15m": 15, "60m": 60}.items():
        rows: list[dict[str, object]] = []
        for session in ("2026-05-25", "2026-05-26"):
            for index in range(4):
                close = 11.0 + index * 0.05
                rows.append(
                    {
                        "date": pd.Timestamp(f"{session} 09:30:00") + pd.Timedelta(minutes=minutes * index),
                        "stock_code": "000001.SZ",
                        "open": close - 0.02,
                        "high": close + 0.08,
                        "low": close - 0.08,
                        "close": close,
                        "volume": 1000.0,
                        "amount": close * 1000.0,
                    }
                )
        write_local_bars(data_root=data_root, timeframe=timeframe, adjust="qfq", bars=pd.DataFrame(rows))

    result = scan_timeframes(
        data_root=data_root,
        timeframes=("15m", "60m"),
        adjust="qfq",
        symbols=("000001.SZ",),
        start="2026-05-25",
        end="2026-05-26",
        strategy=StrategyConfig(channel_lookback=3, landmark_lookback=3),
    )

    assert result.data_audit["status"].tolist() == ["ok", "ok"]
    assert result.filtered_limit_open_days["session_date"].tolist() == [pd.Timestamp("2026-05-25")]
    assert set(result.full["timeframe"]) == {"15m", "60m"}
    assert result.full["date"].dt.normalize().unique().tolist() == [pd.Timestamp("2026-05-26")]
