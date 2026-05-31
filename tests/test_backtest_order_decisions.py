from __future__ import annotations

import pandas as pd
import pytest

from trending_winning.backtest.execution import OrderExecutionResult
from trending_winning.backtest.order_decisions import (
    order_decision_record,
    order_duplicate_reject_reason,
    order_preflight_reject_reason,
    validate_order_frame_columns,
)


def _order(**overrides: object) -> pd.Series:
    data: dict[str, object] = {
        "order_id": "order-1",
        "event_id": "event-1",
        "event_type": "trend_signal_bar",
        "strategy_name": "trend_signal_bar",
        "detector_name": "trend",
        "stock_code": "000001.SZ",
        "timeframe": "5m",
        "signal_date": pd.Timestamp("2026-05-25 09:35:00"),
        "signal_bar_index": 1,
        "side": "long",
        "entry_price": 10.0,
        "stop_price": 9.5,
        "target_price": 11.0,
    }
    data.update(overrides)
    return pd.Series(data)


def test_order_frame_validation_reports_missing_fields() -> None:
    orders = pd.DataFrame([_order().drop(labels=["target_price"])])

    with pytest.raises(ValueError, match="target_price"):
        validate_order_frame_columns(orders)


def test_order_duplicate_reject_reason_tracks_seen_ids() -> None:
    seen: set[str] = set()

    assert order_duplicate_reject_reason(_order(), seen) == ""
    assert order_duplicate_reject_reason(_order(), seen) == "duplicate_order_id"
    assert order_duplicate_reject_reason(_order(order_id=""), seen) == ""


def test_order_preflight_reject_reason_validates_direction_prices_and_target() -> None:
    assert order_preflight_reject_reason(_order()) == ""
    assert order_preflight_reject_reason(_order(side="long", target_price=9.8)) == "target_not_favorable"
    assert order_preflight_reject_reason(_order(signal_bar_index=-1)) == "invalid_order"
    assert order_preflight_reject_reason(_order(stock_code="")) == "invalid_order"


def test_order_decision_record_uses_execution_metrics_and_trade_dates() -> None:
    order = _order(_portfolio_priority=3)
    trade = {
        "order_id": "order-1",
        "event_id": "event-1",
        "side": "long",
        "entry_date": pd.Timestamp("2026-05-25 09:40:00"),
        "entry_price": 10.2,
        "stop_price": 9.5,
        "target_price": 11.0,
    }
    execution = OrderExecutionResult(
        trade=trade,
        actual_entry_price=10.2,
        actual_risk_pct=0.06862745098,
        actual_chase_pct=0.02,
        actual_reward_to_risk=1.142857142857,
    )

    record = order_decision_record(
        order,
        "accepted",
        "",
        trade=trade,
        capital_fraction=0.4,
        risk_fraction=0.02,
        margin_fraction=0.4,
        sector="银行",
        execution=execution,
    )

    assert record["status"] == "accepted"
    assert record["entry_date"] == pd.Timestamp("2026-05-25 09:40:00")
    assert record["actual_entry_price"] == pytest.approx(10.2)
    assert record["actual_chase_pct"] == pytest.approx(0.02)
    assert record["portfolio_priority"] == 3
    assert record["sector"] == "银行"
