from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from trending_winning.strategies.diagnostics import empty_strategy_filter_decisions


ORDER_DECISION_COLUMNS = [
    "order_id",
    "event_id",
    "event_type",
    "strategy_name",
    "detector_name",
    "stock_code",
    "timeframe",
    "signal_date",
    "signal_bar_index",
    "side",
    "planned_entry_price",
    "entry_date",
    "actual_entry_price",
    "actual_risk_pct",
    "actual_chase_pct",
    "actual_reward_to_risk",
    "status",
    "reason",
    "portfolio_priority",
    "capital_fraction",
    "risk_fraction",
    "margin_fraction",
    "sector",
]


ORDER_REQUIRED_COLUMNS = frozenset(
    {
        "order_id",
        "event_id",
        "stock_code",
        "signal_date",
        "signal_bar_index",
        "side",
        "entry_price",
        "stop_price",
        "target_price",
    }
)


TRADE_COLUMNS = [
    "order_id",
    "event_id",
    "event_type",
    "strategy_name",
    "detector_name",
    "stock_code",
    "timeframe",
    "signal_date",
    "signal_bar_index",
    "side",
    "planned_entry_price",
    "entry_date",
    "entry_price",
    "stop_price",
    "target_price",
    "risk_per_share",
    "exit_date",
    "exit_price",
    "exit_reason",
    "holding_bars",
    "return_pct",
    "r_multiple",
    "mae_pct",
    "mfe_pct",
    "mae_r",
    "mfe_r",
    "metadata",
]


@dataclass(frozen=True)
class BacktestConfig:
    """回测撮合参数；只描述入场后的止盈、止损、费用和初始资金。"""

    take_profit_pct: float = 0.06
    stop_loss_pct: float = 0.03
    max_holding_bars: int = 12
    fee_rate: float = 0.0
    slippage_bps: float = 0.0
    initial_equity: float = 1.0
    intrabar_exit_policy: str = "conservative"
    trailing_take_profit_activation_pct: float = 0.0
    trailing_take_profit_drawdown_pct: float = 0.0
    trailing_take_profit_ma_period: int = 0


@dataclass(frozen=True)
class BacktestResult:
    """回测输出对象；逐笔交易、净值曲线和绩效统计分开保存。"""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: dict[str, object]
    order_decisions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ORDER_DECISION_COLUMNS))
    strategy_filter_decisions: pd.DataFrame = field(default_factory=empty_strategy_filter_decisions)
