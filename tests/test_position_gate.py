from __future__ import annotations

import pandas as pd

from trending_winning.backtest.position_gate import apply_single_position_gate


def _candidate(order_sequence: int, order_id: str, entry: str, exit_: str) -> dict[str, object]:
    return {
        "order_sequence": order_sequence,
        "trade": {
            "order_id": order_id,
            "stock_code": order_id[-1],
            "entry_date": pd.Timestamp(entry),
            "exit_date": pd.Timestamp(exit_),
        },
    }


def test_single_position_gate_uses_actual_entry_time_and_rejects_overlap() -> None:
    decisions = apply_single_position_gate(
        [
            _candidate(0, "late-a", "2026-05-25 10:30:00", "2026-05-25 11:00:00"),
            _candidate(1, "early-b", "2026-05-25 10:00:00", "2026-05-25 10:45:00"),
            _candidate(2, "overlap-c", "2026-05-25 10:30:00", "2026-05-25 11:30:00"),
            _candidate(3, "after-d", "2026-05-25 11:30:00", "2026-05-25 12:00:00"),
        ]
    )

    assert [(item.candidate["trade"]["order_id"], item.status, item.reason) for item in decisions] == [
        ("early-b", "accepted", ""),
        ("late-a", "rejected", "already_open"),
        ("overlap-c", "rejected", "already_open"),
        ("after-d", "accepted", ""),
    ]
