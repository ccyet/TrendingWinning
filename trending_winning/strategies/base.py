from __future__ import annotations

from typing import Protocol

import pandas as pd


ORDER_COLUMNS = [
    "order_id",
    "strategy_name",
    "detector_name",
    "event_id",
    "event_type",
    "stock_code",
    "timeframe",
    "signal_date",
    "signal_bar_index",
    "side",
    "signal_price",
    "entry_price",
    "stop_price",
    "target_price",
    "max_holding_bars",
    "max_actual_risk_pct",
    "max_chase_pct",
    "metadata",
]


class Strategy(Protocol):
    """策略协议；单策略回测只依赖本策略绑定的识别器。"""

    name: str

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        ...


def empty_orders() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(ORDER_COLUMNS))
