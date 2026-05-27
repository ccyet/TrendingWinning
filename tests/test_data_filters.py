from __future__ import annotations

import pandas as pd

from trending_winning.data.filters import board_limit_pct, filter_limit_open_days


def test_board_limit_pct_uses_a_share_board_rules() -> None:
    assert board_limit_pct("600519.SH") == 0.10
    assert board_limit_pct("000001.SZ") == 0.10
    assert board_limit_pct("688001.SH") == 0.20
    assert board_limit_pct("300750.SZ") == 0.20
    assert board_limit_pct("920001.BJ") == 0.30


def test_filter_limit_open_days_removes_intraday_bars_for_limit_open_day() -> None:
    intraday = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 09:35:00",
                    "2026-05-25 09:40:00",
                    "2026-05-26 09:35:00",
                ]
            ),
            "stock_code": ["688001.SH", "688001.SH", "688001.SH"],
            "open": [12.0, 12.1, 12.2],
            "high": [12.2, 12.2, 12.4],
            "low": [11.9, 12.0, 12.0],
            "close": [12.1, 12.0, 12.3],
            "volume": [1000.0, 1000.0, 1000.0],
            "amount": [12100.0, 12000.0, 12300.0],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-24", "2026-05-25", "2026-05-26"]),
            "stock_code": ["688001.SH", "688001.SH", "688001.SH"],
            "open": [10.0, 12.0, 12.1],
            "high": [10.1, 12.2, 12.4],
            "low": [9.8, 11.9, 12.0],
            "close": [10.0, 12.0, 12.3],
            "volume": [1000.0, 3000.0, 2800.0],
            "amount": [10000.0, 36000.0, 34440.0],
        }
    )

    filtered = filter_limit_open_days(intraday, daily)

    assert filtered["date"].dt.normalize().unique().tolist() == [pd.Timestamp("2026-05-26")]
