from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.portfolio_equity import build_portfolio_equity_curve_from_normalized
from trending_winning.data.schema import normalize_bars


def test_portfolio_equity_curve_marks_short_cash_value_and_margin_exposure() -> None:
    bars = normalize_bars(
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00"),
                    "stock_code": "000001.SZ",
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 10.0,
                    "volume": 1000.0,
                    "amount": 10000.0,
                },
                {
                    "date": pd.Timestamp("2026-05-25 10:00:00"),
                    "stock_code": "000001.SZ",
                    "open": 9.0,
                    "high": 9.0,
                    "low": 9.0,
                    "close": 9.0,
                    "volume": 1000.0,
                    "amount": 9000.0,
                },
            ]
        )
    )
    trades = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-05-25 09:30:00"),
                "exit_date": pd.Timestamp("2026-05-25 10:00:00"),
                "stock_code": "000001.SZ",
                "side": "short",
                "entry_price": 10.0,
                "raw_return_pct": 10.0,
                "capital_fraction": 0.5,
                "margin_fraction": 1.0,
                "portfolio_priority": 1,
            }
        ]
    )

    equity = build_portfolio_equity_curve_from_normalized(bars, trades, initial_equity=1.0)

    open_row = equity.iloc[0]
    assert open_row["cash"] == pytest.approx(1.5)
    assert open_row["position_value"] == pytest.approx(-0.5)
    assert open_row["net_value"] == pytest.approx(1.0)
    assert open_row["gross_exposure"] == pytest.approx(0.5)
    assert open_row["margin_exposure"] == pytest.approx(1.0)
    assert equity.iloc[-1]["cash"] == pytest.approx(1.05)
    assert equity.iloc[-1]["net_value"] == pytest.approx(1.05)
