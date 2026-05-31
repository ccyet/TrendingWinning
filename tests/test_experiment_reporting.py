from __future__ import annotations

import pandas as pd

from trending_winning.backtest.reporting import (
    detector_trade_statistics,
    setup_trade_statistics,
    strategy_trade_statistics,
    trade_path_distribution_statistics,
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


def test_trade_path_distribution_statistics_bucket_trade_quality() -> None:
    trades = pd.DataFrame(
        {
            "return_pct": [3.0, -1.0, 5.0, -4.0, 8.0],
            "holding_bars": [1, 2, 5, 10, 20],
            "r_multiple": [0.8, -0.5, 1.5, -1.2, 2.5],
            "mae_r": [-0.2, -0.8, -0.4, -1.1, -0.3],
            "mfe_r": [1.0, 0.2, 2.0, 0.1, 3.0],
        }
    )

    stats = trade_path_distribution_statistics(trades)

    assert set(stats["dimension"]) == {"持有K数", "R倍数", "最大不利R", "最大有利R"}
    by_bucket = stats.set_index(["dimension", "bucket"])
    assert by_bucket.loc[("R倍数", "1R~2R"), "trade_count"] == 1.0
    assert by_bucket.loc[("R倍数", "1R~2R"), "avg_return"] == 0.05
    assert by_bucket.loc[("R倍数", "<=-1R"), "win_rate"] == 0.0
    assert by_bucket.loc[("持有K数", "9-16K"), "trade_count"] == 1.0
    assert by_bucket.loc[("最大有利R", ">=2R"), "trade_count"] == 2.0


def test_trade_path_distribution_statistics_returns_stable_empty_frame() -> None:
    stats = trade_path_distribution_statistics(pd.DataFrame())

    assert stats.columns.tolist() == [
        "dimension",
        "bucket",
        "bucket_order",
        "trade_count",
        "win_rate",
        "avg_return",
        "avg_r_multiple",
        "avg_mae_r",
        "avg_mfe_r",
        "avg_holding_bars",
    ]
    assert stats.empty
