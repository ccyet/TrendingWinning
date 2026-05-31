from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.trade_path import (
    equity_with_initial_point,
    trade_path_frame,
    trade_returns_as_decimal,
)


def test_trade_path_frame_sorts_by_realized_time_trade_number_and_input_order() -> None:
    trades = pd.DataFrame(
        {
            "trade_no": [3, 2, 1, 4],
            "exit_date": [
                "2026-05-25 10:10:00",
                "2026-05-25 10:05:00",
                "2026-05-25 10:05:00",
                "2026-05-25 10:10:00",
            ],
            "return_pct": [3.0, 2.0, 1.0, 4.0],
        }
    )

    result = trade_path_frame(trades)

    assert result["trade_no"].tolist() == [1, 2, 3, 4]
    assert "_realized_at" not in result.columns
    assert "_row_order" not in result.columns


def test_trade_path_frame_keeps_input_order_when_time_axis_is_missing() -> None:
    trades = pd.DataFrame({"return_pct": [2.0, -1.0, 3.0]})

    result = trade_path_frame(trades)

    assert result["return_pct"].tolist() == [2.0, -1.0, 3.0]


def test_trade_returns_and_equity_with_initial_point_use_decimal_path() -> None:
    returns = trade_returns_as_decimal(pd.DataFrame({"return_pct": [10.0, -5.0]}))
    equity = equity_with_initial_point(returns)

    assert returns.tolist() == pytest.approx([0.1, -0.05])
    assert equity.tolist() == pytest.approx([1.0, 1.1, 1.045])
