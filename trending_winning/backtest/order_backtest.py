from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from trending_winning.backtest.execution import (
    coerce_order_execution_result,
    simulate_order_trade_with_rejection,
    validate_backtest_config,
)
from trending_winning.backtest.models import BacktestConfig, BacktestResult, ORDER_DECISION_COLUMNS, TRADE_COLUMNS
from trending_winning.backtest.order_decisions import (
    order_decision_record,
    order_duplicate_reject_reason,
    order_preflight_reject_reason,
    validate_order_frame_columns,
)
from trending_winning.backtest.position_gate import apply_single_position_gate
from trending_winning.backtest.portfolio_equity import build_single_position_equity_curve_from_normalized
from trending_winning.backtest.stats import (
    build_equity_curve,
    compute_equity_statistics,
    compute_trade_statistics,
    summarize_exit_reasons,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.strategies.diagnostics import empty_strategy_filter_decisions


def run_order_backtest(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return run_order_backtest_from_normalized(normalize_bars(bars), orders, cfg)


def run_order_backtest_from_normalized(
    normalized_bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """基于已标准化 K 线做订单回测；参数遍历热路径用它避免重复 normalize。"""
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return _run_order_backtest_from_normalized(normalized_bars, orders, cfg)


def order_statistics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    market_bar_count: int = 0,
) -> dict[str, object]:
    stats = _trade_statistics(trades, equity_curve)
    stats.update(single_position_exposure_statistics(stats, market_bar_count=market_bar_count))
    stats.update(summarize_order_decisions(decisions))
    stats.update(summarize_strategy_filter_decisions(empty_strategy_filter_decisions()))
    return stats


def single_position_exposure_statistics(stats: Mapping[str, object], *, market_bar_count: int) -> dict[str, float]:
    """单策略是满仓进出，场内比例应按全市场时间轴计算，不能沿用逐 K 组合净值默认值。"""
    market_count = max(int(market_bar_count), 0)
    exposure_bars = max(_as_float(stats.get("exposure_bars", 0.0)), 0.0)
    ratio = min(1.0, exposure_bars / market_count) if market_count > 0 else 0.0
    return {
        "market_bar_count": float(market_count),
        "exposure_bar_ratio": float(ratio),
    }


def market_bar_count(bars: pd.DataFrame) -> int:
    if bars.empty or "date" not in bars.columns:
        return 0
    dates = pd.to_datetime(bars["date"], errors="coerce")
    return int(dates.dropna().nunique())


def trade_risk_fraction(trade: Mapping[str, object]) -> float:
    entry_price = float(trade["entry_price"])
    risk_per_share = float(trade["risk_per_share"])
    if entry_price <= 0 or risk_per_share <= 0:
        return 0.0
    return float(risk_per_share / entry_price)


def sort_trades_for_statistics(trades: pd.DataFrame) -> pd.DataFrame:
    """成交按真实入场时间排序，避免统计结果受股票代码顺序影响。"""
    if trades.empty or "entry_date" not in trades.columns:
        return trades
    result = trades.copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    return result.sort_values(["entry_date", "stock_code", "order_id"]).reset_index(drop=True)


def _run_order_backtest_from_normalized(
    normalized: pd.DataFrame,
    orders: pd.DataFrame,
    cfg: BacktestConfig,
) -> BacktestResult:
    if orders.empty:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        decisions = pd.DataFrame(columns=ORDER_DECISION_COLUMNS)
        equity = build_equity_curve(trades, cfg.initial_equity)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=order_statistics(
                trades,
                equity,
                decisions,
                market_bar_count=market_bar_count(normalized),
            ),
            order_decisions=decisions,
        )

    trades: list[dict[str, object]] = []
    decisions: list[tuple[int, dict[str, object]]] = []
    candidates: list[dict[str, object]] = []
    sorted_orders = _sorted_order_records(orders)
    seen_order_ids: set[str] = set()
    if normalized.empty:
        empty_decisions: list[dict[str, object]] = []
        for order in sorted_orders:
            reason = (
                order_duplicate_reject_reason(order, seen_order_ids)
                or order_preflight_reject_reason(order)
                or "no_bars"
            )
            empty_decisions.append(order_decision_record(order, "rejected", reason))
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades, cfg.initial_equity)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=order_statistics(
                trades,
                equity,
                pd.DataFrame(empty_decisions, columns=ORDER_DECISION_COLUMNS),
                market_bar_count=market_bar_count(normalized),
            ),
            order_decisions=pd.DataFrame(empty_decisions, columns=ORDER_DECISION_COLUMNS),
        )

    bars_by_symbol = {
        symbol: group.reset_index(drop=True)
        for symbol, group in normalized.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False)
    }
    for order_sequence, order in enumerate(sorted_orders):
        duplicate_rejection = order_duplicate_reject_reason(order, seen_order_ids)
        if duplicate_rejection:
            decisions.append((order_sequence, order_decision_record(order, "rejected", duplicate_rejection)))
            continue
        preflight_rejection = order_preflight_reject_reason(order)
        if preflight_rejection:
            decisions.append((order_sequence, order_decision_record(order, "rejected", preflight_rejection)))
            continue
        symbol = normalize_symbol(order["stock_code"])
        group = bars_by_symbol.get(symbol)
        if group is None or group.empty:
            decisions.append((order_sequence, order_decision_record(order, "rejected", "no_bars")))
            continue
        signal_index = int(order["signal_bar_index"])
        execution = coerce_order_execution_result(
            simulate_order_trade_with_rejection(group, order, signal_index, cfg),
            order=order,
        )
        trade = execution.trade
        if trade is None:
            decisions.append(
                (
                    order_sequence,
                    order_decision_record(
                        order,
                        "rejected",
                        execution.reject_reason or "no_fill",
                        execution=execution,
                    ),
                )
            )
            continue
        candidates.append(
            {
                "order_sequence": order_sequence,
                "order": order,
                "trade": trade,
                "execution": execution,
            }
        )

    for gate_decision in apply_single_position_gate(candidates):
        candidate = gate_decision.candidate
        order_sequence = int(candidate["order_sequence"])
        order = candidate["order"]
        trade = candidate["trade"]
        execution = candidate["execution"]
        if gate_decision.status == "rejected":
            decisions.append((order_sequence, order_decision_record(order, "rejected", gate_decision.reason, execution=execution)))
            continue
        trades.append(trade)
        decisions.append(
            (
                order_sequence,
                order_decision_record(
                    order,
                    "accepted",
                    "",
                    trade=trade,
                    execution=execution,
                    capital_fraction=1.0,
                    risk_fraction=trade_risk_fraction(trade),
                    margin_fraction=1.0,
                ),
            )
        )

    decision_records = [record for _, record in sorted(decisions, key=lambda item: item[0])]
    decisions_df = pd.DataFrame(decision_records, columns=ORDER_DECISION_COLUMNS)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades_df, cfg.initial_equity)
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity,
            stats=order_statistics(
                trades_df,
                equity,
                decisions_df,
                market_bar_count=market_bar_count(normalized),
            ),
            order_decisions=decisions_df,
        )

    trades_df = sort_trades_for_statistics(trades_df.drop(columns=["_exit_index"]))
    trades_df = trades_df[TRADE_COLUMNS]
    equity = build_single_position_equity_curve_from_normalized(normalized, trades_df, cfg.initial_equity)
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity,
        stats=order_statistics(
            trades_df,
            equity,
            decisions_df,
            market_bar_count=market_bar_count(normalized),
        ),
        order_decisions=decisions_df,
    )


def _trade_statistics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, object]:
    """合并逐笔统计和净值曲线统计；净值指标以初始资金点为基准。"""
    stats = compute_trade_statistics(trades)
    stats.update(summarize_exit_reasons(trades))
    stats.update(compute_equity_statistics(equity_curve))
    return stats


def _sort_orders_for_execution(orders: pd.DataFrame) -> pd.DataFrame:
    validate_order_frame_columns(orders)
    result = orders.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce")
    return result.sort_values(["signal_date", "stock_code", "order_id"]).reset_index(drop=True)


def _sorted_order_records(orders: pd.DataFrame) -> list[dict[str, object]]:
    """按撮合顺序输出订单记录，避免热路径反复创建 Series 行对象。"""
    return _sort_orders_for_execution(orders).to_dict("records")


def _as_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
