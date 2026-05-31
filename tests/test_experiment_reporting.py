from __future__ import annotations

import pandas as pd

from trending_winning.backtest.reporting import (
    detector_trade_statistics,
    setup_trade_statistics,
    strategy_trade_statistics,
    trade_dated_equity_curve,
)


class _NamedStrategy:
    def __init__(self, name: str) -> None:
        self.name = name


def test_reporting_module_preserves_zero_rows_for_enabled_strategy_detector_and_setup() -> None:
    trades = pd.DataFrame(
        [
            {
                "strategy_name": "trend_signal_bar",
                "detector_name": "trend",
                "event_type": "h2_pullback",
                "side": "long",
                "stock_code": "000001.SZ",
                "exit_reason": "take_profit",
                "return_pct": 3.0,
                "entry_date": pd.Timestamp("2026-05-25 10:00"),
                "exit_date": pd.Timestamp("2026-05-25 11:00"),
            }
        ]
    )
    order_decisions = pd.DataFrame(
        [
            {
                "strategy_name": "range_signal_bar",
                "detector_name": "range",
                "event_type": "failed_breakout",
                "side": "short",
                "status": "rejected",
                "reason": "no_fill",
            }
        ]
    )

    strategy_stats = strategy_trade_statistics(
        trades,
        [_NamedStrategy("trend_signal_bar"), _NamedStrategy("range_signal_bar")],
        order_decisions=order_decisions,
    ).set_index("strategy_name")
    detector_stats = detector_trade_statistics(
        trades,
        ("trend", "range"),
        order_decisions=order_decisions,
    ).set_index("detector_name")
    setup_stats = setup_trade_statistics(trades, order_decisions=order_decisions).set_index(
        ["detector_name", "event_type", "side"]
    )

    assert strategy_stats.loc["trend_signal_bar", "trade_count"] == 1.0
    assert strategy_stats.loc["range_signal_bar", "trade_count"] == 0.0
    assert detector_stats.loc["trend", "trade_count"] == 1.0
    assert detector_stats.loc["range", "trade_count"] == 0.0
    assert setup_stats.loc[("range", "failed_breakout", "short"), "trade_count"] == 0.0


def test_trade_dated_equity_curve_uses_exit_dates_without_backtest_result_dependency() -> None:
    equity = pd.DataFrame({"trade_no": [0, 1, 2], "net_value": [1.0, 1.03, 1.01]})
    trades = pd.DataFrame(
        {
            "exit_date": [
                pd.Timestamp("2026-05-25 11:00"),
                pd.Timestamp("2026-05-26 10:00"),
            ]
        }
    )

    dated = trade_dated_equity_curve(equity, trades)

    assert dated["date"].tolist() == [
        pd.Timestamp("2026-05-25 11:00"),
        pd.Timestamp("2026-05-25 11:00"),
        pd.Timestamp("2026-05-26 10:00"),
    ]
