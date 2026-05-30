from __future__ import annotations

from inspect import getsource

import pandas as pd
import pytest

from trending_winning.backtest.engine import BacktestConfig
from trending_winning.backtest.execution import simulate_order_trade
from trending_winning.backtest.engine import run_order_backtest


def _bars(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volume", 1000.0),
                "amount": row.get("amount", row["close"] * 1000.0),
            }
            for index, row in enumerate(rows)
        ]
    )


def _order(*, side: str, entry_price: float, stop_price: float, target_price: float) -> pd.Series:
    return pd.Series(
        {
            "strategy_name": "execution_case",
            "detector_name": "trend",
            "stock_code": "000001.SZ",
            "side": side,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "max_holding_bars": 2,
        }
    )


def test_order_backtest_rejects_gap_fill_when_actual_risk_exceeds_order_limit() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 11.0, "high": 11.2, "low": 10.9, "close": 11.1},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "gap-risk",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-risk",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.9,
                "target_price": 10.8,
                "max_holding_bars": 1,
                "max_actual_risk_pct": 0.05,
                "max_chase_pct": 0.2,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "actual_risk_too_high"
    assert decision["actual_entry_price"] == pytest.approx(11.0)
    assert decision["actual_risk_pct"] == pytest.approx((11.0 - 9.9) / 11.0)
    assert decision["actual_chase_pct"] == pytest.approx(0.1)


def test_order_backtest_rejects_gap_fill_when_actual_chase_exceeds_order_limit() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 11.0, "high": 11.2, "low": 10.9, "close": 11.1},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "gap-chase",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-chase",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.9,
                "target_price": 10.8,
                "max_holding_bars": 1,
                "max_actual_risk_pct": 0.2,
                "max_chase_pct": 0.05,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "chase_too_far"
    assert decision["actual_entry_price"] == pytest.approx(11.0)
    assert decision["actual_risk_pct"] == pytest.approx((11.0 - 9.9) / 11.0)
    assert decision["actual_chase_pct"] == pytest.approx(0.1)


def test_order_backtest_records_actual_execution_metrics_for_accepted_gap_fill() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 11.0, "high": 12.2, "low": 10.9, "close": 11.8},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "gap-accepted",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-accepted",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.9,
                "target_price": 12.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": 0.2,
                "max_chase_pct": 0.2,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.loc[0, "entry_price"] == pytest.approx(11.0)
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "accepted"
    assert decision["actual_entry_price"] == pytest.approx(11.0)
    assert decision["actual_risk_pct"] == pytest.approx((11.0 - 9.9) / 11.0)
    assert decision["actual_chase_pct"] == pytest.approx(0.1)
    assert decision["actual_reward_to_risk"] == pytest.approx((12.0 - 11.0) / (11.0 - 9.9))


@pytest.mark.parametrize(
    ("side", "expected_side", "entry_price", "stop_price", "target_price"),
    [
        (" Long ", "long", 10.2, 9.8, 11.0),
        (" SHORT ", "short", 9.8, 10.2, 9.0),
    ],
)
def test_order_backtest_normalizes_side_text_at_execution_boundary(
    side: str,
    expected_side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.0, "high": 10.6, "low": 9.4, "close": 10.1},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": f"side-{expected_side}",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": f"event-side-{expected_side}",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": side,
                "signal_price": 10.0,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades["side"].tolist() == [expected_side]
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "accepted"
    assert decision["side"] == expected_side


def test_order_backtest_rejects_duplicate_order_ids_after_first_occurrence() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.2, "high": 11.2, "low": 10.1, "close": 10.8},
            {"open": 10.8, "high": 11.4, "low": 10.7, "close": 11.1},
        ]
    )
    base_order = {
        "order_id": "duplicate-order",
        "strategy_name": "execution_case",
        "detector_name": "trend",
        "stock_code": "000001.SZ",
        "timeframe": "30m",
        "signal_date": bars.loc[0, "date"],
        "signal_bar_index": 0,
        "side": "long",
        "signal_price": 10.0,
        "entry_price": 10.2,
        "stop_price": 9.8,
        "target_price": 11.2,
        "max_holding_bars": 1,
        "max_actual_risk_pct": None,
        "max_chase_pct": None,
        "metadata": {},
    }
    orders = pd.DataFrame(
        [
            {**base_order, "event_id": "event-first"},
            {**base_order, "event_id": "event-duplicate"},
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades["order_id"].tolist() == ["duplicate-order"]
    assert result.order_decisions["status"].tolist() == ["accepted", "rejected"]
    assert result.order_decisions["reason"].tolist() == ["", "duplicate_order_id"]
    assert result.order_decisions["event_id"].tolist() == ["event-first", "event-duplicate"]
    assert result.stats["rejected_duplicate_order_id_count"] == 1.0


@pytest.mark.parametrize(
    ("side", "entry_price", "stop_price", "target_price"),
    [
        ("long", 10.2, 10.4, 11.0),
        ("short", 10.2, 10.0, 9.4),
    ],
)
def test_order_backtest_rejects_non_protective_stop_before_simulation(
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.4, "low": 9.8, "close": 10.0},
            {"open": 10.2, "high": 10.6, "low": 9.6, "close": 10.1},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-stop",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-stop",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": side,
                "signal_price": 10.0,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert decision["actual_entry_price"] == 0.0
    assert result.stats["rejected_invalid_order_count"] == 1.0


@pytest.mark.parametrize("order_id", ["", "   ", None])
def test_order_backtest_rejects_orders_without_traceable_order_id(order_id: object) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-missing-order-id",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


@pytest.mark.parametrize("event_id", ["", "   ", None])
def test_order_backtest_rejects_orders_without_traceable_event_id(event_id: object) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "missing-event-id",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": event_id,
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_order_backtest_prefers_invalid_order_over_no_bars_when_market_data_is_empty() -> None:
    bars = pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-empty-market-order",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0
    assert result.stats["rejected_no_bars_count"] == 0.0


@pytest.mark.parametrize(
    ("side", "entry_price", "stop_price", "target_price"),
    [
        ("long", 10.2, 9.8, 10.0),
        ("short", 9.8, 10.2, 10.0),
    ],
)
def test_order_backtest_prefers_target_direction_rejection_over_no_bars_when_market_data_is_empty(
    side: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> None:
    bars = pd.DataFrame(columns=["date", "stock_code", "open", "high", "low", "close", "volume", "amount"])
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-target-empty-market",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-target-empty-market",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                "signal_bar_index": 0,
                "side": side,
                "signal_price": 10.0,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "target_not_favorable"
    assert result.stats["rejected_target_not_favorable_count"] == 1.0
    assert result.stats["rejected_no_bars_count"] == 0.0


def test_order_backtest_reports_missing_required_order_columns_clearly() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "missing-signal-date",
                "event_id": "event-missing-signal-date",
                "stock_code": "000001.SZ",
                "signal_bar_index": 0,
                "side": "long",
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="订单缺少必要字段：signal_date"):
        run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))


def test_order_backtest_rejects_orders_with_invalid_signal_date() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-date",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-date",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": "not-a-date",
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_order_backtest_rejects_orders_with_non_numeric_signal_bar_index() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-index-text",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-index-text",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": "bad-index",
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert decision["signal_bar_index"] == -1
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_order_backtest_rejects_orders_with_non_numeric_price_fields() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-price",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-price",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": "bad-price",
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert decision["planned_entry_price"] == 0.0
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_order_backtest_rejects_zero_risk_orders_before_execution() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.8, "low": 10.0, "close": 10.6},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "zero-risk",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-zero-risk",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 10.2,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


def test_order_backtest_uses_config_max_holding_when_order_value_is_missing() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.6, "low": 10.0, "close": 10.5},
            {"open": 10.5, "high": 11.0, "low": 10.4, "close": 10.9},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "missing-max-holding",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-missing-max-holding",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.5,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=2))

    assert result.order_decisions.iloc[0]["status"] == "accepted"
    assert result.trades.loc[0, "order_id"] == "missing-max-holding"
    assert result.trades.loc[0, "holding_bars"] == 1
    assert result.trades.loc[0, "exit_reason"] == "max_holding"


def test_order_backtest_rejects_non_numeric_max_holding_bars() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.6, "low": 10.0, "close": 10.5},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "bad-max-holding",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-bad-max-holding",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.5,
                "max_holding_bars": "bad",
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=2))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "invalid_order"
    assert result.stats["rejected_invalid_order_count"] == 1.0


@pytest.mark.parametrize(
    ("signal_bar_index", "expected_reason"),
    [(-1, "invalid_order"), (5, "no_bars")],
)
def test_order_backtest_rejects_orders_with_invalid_signal_bar_index(
    signal_bar_index: int,
    expected_reason: str,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.1, "high": 10.6, "low": 10.0, "close": 10.5},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": f"bad-index-{signal_bar_index}",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": f"event-index-{signal_bar_index}",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": signal_bar_index,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == expected_reason
    assert decision["actual_entry_price"] == 0.0


def test_order_backtest_rejects_entry_on_zero_liquidity_bar() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.4, "high": 10.8, "low": 10.3, "close": 10.6, "volume": 0.0, "amount": 0.0},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": "zero-liquidity-entry",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": "event-zero-liquidity-entry",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": "long",
                "signal_price": 10.0,
                "entry_price": 10.2,
                "stop_price": 9.8,
                "target_price": 11.0,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "no_liquidity"
    assert decision["actual_entry_price"] == 0.0
    assert result.stats["rejected_no_liquidity_count"] == 1.0


@pytest.mark.parametrize(
    ("side", "open_price", "high", "low", "entry_price", "stop_price", "target_price"),
    [
        ("long", 11.0, 11.2, 10.9, 10.2, 9.8, 10.8),
        ("short", 9.0, 9.2, 8.8, 9.8, 10.2, 9.2),
    ],
)
def test_order_backtest_rejects_gap_fill_when_target_is_no_longer_favorable(
    side: str,
    open_price: float,
    high: float,
    low: float,
    entry_price: float,
    stop_price: float,
    target_price: float,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": open_price, "high": high, "low": low, "close": open_price},
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "order_id": f"gap-target-{side}",
                "strategy_name": "execution_case",
                "detector_name": "trend",
                "event_id": f"event-target-{side}",
                "stock_code": "000001.SZ",
                "timeframe": "30m",
                "signal_date": bars.loc[0, "date"],
                "signal_bar_index": 0,
                "side": side,
                "signal_price": 10.0,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "max_holding_bars": 1,
                "max_actual_risk_pct": None,
                "max_chase_pct": None,
                "metadata": {},
            }
        ]
    )

    result = run_order_backtest(bars, orders, BacktestConfig(max_holding_bars=1))

    assert result.trades.empty
    decision = result.order_decisions.iloc[0]
    assert decision["status"] == "rejected"
    assert decision["reason"] == "target_not_favorable"
    assert decision["actual_entry_price"] == pytest.approx(open_price)
    assert decision["actual_reward_to_risk"] <= 0.0
    assert result.stats["rejected_target_not_favorable_count"] == 1.0


@pytest.mark.parametrize(
    ("side", "entry_price", "open_price", "expected_fill"),
    [
        ("long", 10.5, 10.9, 10.9),
        ("short", 9.5, 9.1, 9.1),
    ],
)
def test_stop_entry_order_fills_at_open_when_gap_crosses_trigger(
    side: str,
    entry_price: float,
    open_price: float,
    expected_fill: float,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": open_price, "high": max(open_price, 11.2), "low": min(open_price, 8.8), "close": open_price},
        ]
    )

    trade = simulate_order_trade(
        bars,
        _order(
            side=side,
            entry_price=entry_price,
            stop_price=9.8 if side == "long" else 10.2,
            target_price=11.5 if side == "long" else 8.5,
        ),
        signal_index=0,
        cfg=BacktestConfig(max_holding_bars=1),
    )

    assert trade is not None
    assert trade["entry_price"] == pytest.approx(expected_fill)


@pytest.mark.parametrize(
    ("side", "exit_open", "expected_exit", "expected_reason"),
    [
        ("long", 9.4, 9.4, "stop_loss"),
        ("long", 11.8, 11.8, "take_profit"),
        ("short", 10.6, 10.6, "stop_loss"),
        ("short", 8.2, 8.2, "take_profit"),
    ],
)
def test_exit_order_fills_at_open_when_gap_crosses_stop_or_target(
    side: str,
    exit_open: float,
    expected_exit: float,
    expected_reason: str,
) -> None:
    if side == "long" and expected_reason == "stop_loss":
        exit_bar = {"open": exit_open, "high": 9.6, "low": 9.2, "close": exit_open}
    elif side == "long":
        exit_bar = {"open": exit_open, "high": 12.0, "low": 11.6, "close": exit_open}
    elif expected_reason == "stop_loss":
        exit_bar = {"open": exit_open, "high": 10.8, "low": 10.4, "close": exit_open}
    else:
        exit_bar = {"open": exit_open, "high": 8.4, "low": 8.0, "close": exit_open}
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            exit_bar,
        ]
    )

    trade = simulate_order_trade(
        bars,
        _order(
            side=side,
            entry_price=10.0,
            stop_price=9.5 if side == "long" else 10.5,
            target_price=11.5 if side == "long" else 8.5,
        ),
        signal_index=0,
        cfg=BacktestConfig(max_holding_bars=2),
    )

    assert trade is not None
    assert trade["exit_reason"] == expected_reason
    assert trade["exit_price"] == pytest.approx(expected_exit)


def test_order_execution_ignores_zero_liquidity_exit_bar() -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.0, "high": 12.0, "low": 9.0, "close": 10.0, "volume": 0.0, "amount": 0.0},
            {"open": 10.3, "high": 10.5, "low": 10.2, "close": 10.4},
        ]
    )

    trade = simulate_order_trade(
        bars,
        _order(side="long", entry_price=10.0, stop_price=9.5, target_price=11.5),
        signal_index=0,
        cfg=BacktestConfig(max_holding_bars=3),
    )

    assert trade is not None
    assert trade["exit_reason"] == "max_holding"
    assert trade["exit_date"] == pd.Timestamp("2026-05-25 11:00:00")
    assert trade["exit_price"] == pytest.approx(10.4)
    assert trade["mfe_pct"] == pytest.approx(5.0)
    assert trade["mae_pct"] == pytest.approx(-2.0)


def test_order_execution_uses_vectorized_exit_scan_not_cursor_loop() -> None:
    source = getsource(simulate_order_trade)

    assert "for cursor in range" not in source


def test_order_backtest_uses_record_iteration_not_dataframe_iterrows() -> None:
    source = getsource(run_order_backtest)

    assert ".iterrows(" not in source


@pytest.mark.parametrize(
    ("side", "policy", "expected_reason", "expected_exit"),
    [
        ("long", "conservative", "stop_loss", 9.5),
        ("long", "optimistic", "take_profit", 11.5),
        ("short", "conservative", "stop_loss", 10.5),
        ("short", "optimistic", "take_profit", 8.5),
    ],
)
def test_intrabar_exit_policy_resolves_stop_and_target_hit_in_same_bar(
    side: str,
    policy: str,
    expected_reason: str,
    expected_exit: float,
) -> None:
    bars = _bars(
        [
            {"open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"open": 10.0, "high": 11.8, "low": 8.2, "close": 10.0},
        ]
    )

    trade = simulate_order_trade(
        bars,
        _order(
            side=side,
            entry_price=10.0,
            stop_price=9.5 if side == "long" else 10.5,
            target_price=11.5 if side == "long" else 8.5,
        ),
        signal_index=0,
        cfg=BacktestConfig(max_holding_bars=1, intrabar_exit_policy=policy),
    )

    assert trade is not None
    assert trade["exit_reason"] == expected_reason
    assert trade["exit_price"] == pytest.approx(expected_exit)
