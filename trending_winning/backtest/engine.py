from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trending_winning.backtest.execution import (
    OrderExecutionResult,
    coerce_order_execution_result,
    compute_order_execution_metrics,
    simulate_order_trade_with_rejection,
    trade_path_metrics,
    validate_backtest_config,
)
from trending_winning.backtest.stats import (
    build_equity_curve,
    compute_equity_statistics,
    compute_trade_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.strategies.base import Strategy
from trending_winning.strategies.diagnostics import collect_strategy_filter_decisions, empty_strategy_filter_decisions


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


@dataclass(frozen=True)
class BacktestResult:
    """回测输出对象；逐笔交易、净值曲线和绩效统计分开保存。"""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: dict[str, float]
    order_decisions: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=ORDER_DECISION_COLUMNS))
    strategy_filter_decisions: pd.DataFrame = field(default_factory=empty_strategy_filter_decisions)


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


def run_backtest(scanned_bars: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)

    trades: list[dict[str, object]] = []
    for symbol, group in scanned_bars.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False):
        group = group.reset_index(drop=True)
        index = 0
        while index < len(group) - 1:
            if not bool(group.loc[index, "breakout_trigger"]):
                index += 1
                continue
            trade = _simulate_trade(group, index, str(symbol), cfg)
            if trade is None:
                index += 1
                continue
            trades.append(trade)
            index = int(trade["_exit_index"]) + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades_df, cfg.initial_equity)
        return BacktestResult(trades=trades_df, equity_curve=equity, stats=_trade_statistics(trades_df, equity))

    trades_df = _sort_trades_for_statistics(trades_df.drop(columns=["_exit_index"]))
    trades_df = trades_df[TRADE_COLUMNS]
    equity = build_equity_curve(trades_df, cfg.initial_equity)
    return BacktestResult(trades=trades_df, equity_curve=equity, stats=_trade_statistics(trades_df, equity))


def _simulate_trade(
    group: pd.DataFrame,
    entry_index: int,
    symbol: str,
    cfg: BacktestConfig,
) -> dict[str, object] | None:
    entry = group.loc[entry_index]
    entry_price = float(entry["trigger_price"] if pd.notna(entry["trigger_price"]) else entry["close"])
    if entry_price <= 0:
        return None
    target_price = entry_price * (1.0 + cfg.take_profit_pct)
    stop_price = entry_price * (1.0 - cfg.stop_loss_pct)
    last_exit_index = min(len(group) - 1, entry_index + cfg.max_holding_bars)
    if last_exit_index <= entry_index:
        return None

    exit_index, exit_price, exit_reason = _first_legacy_long_exit(
        group,
        first_exit_index=entry_index + 1,
        last_exit_index=last_exit_index,
        stop_price=stop_price,
        target_price=target_price,
    )

    return_pct = (exit_price / entry_price - 1.0) * 100.0
    path_metrics = trade_path_metrics(
        group,
        side="long",
        entry_index=entry_index,
        exit_index=exit_index,
        entry_price=entry_price,
        stop_price=stop_price,
        return_pct=return_pct,
    )
    return {
        "order_id": f"legacy_breakout:{symbol}:{pd.Timestamp(entry['date']).isoformat()}",
        "event_id": f"legacy_scan:{symbol}:{pd.Timestamp(entry['date']).isoformat()}",
        "event_type": "legacy_breakout",
        "strategy_name": "legacy_breakout",
        "detector_name": "legacy_scan",
        "stock_code": symbol,
        "timeframe": "",
        "signal_date": entry["date"],
        "signal_bar_index": int(entry_index),
        "side": "long",
        "planned_entry_price": entry_price,
        "entry_date": entry["date"],
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "risk_per_share": float(abs(entry_price - stop_price)),
        "exit_date": group.loc[exit_index, "date"],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "holding_bars": int(exit_index - entry_index),
        "return_pct": float(return_pct),
        **path_metrics,
        "metadata": {},
        "_exit_index": int(exit_index),
    }


def _first_legacy_long_exit(
    group: pd.DataFrame,
    *,
    first_exit_index: int,
    last_exit_index: int,
    stop_price: float,
    target_price: float,
) -> tuple[int, float, str]:
    """旧版突破回测的长仓退出；保持 stop 优先于 target 的原始语义。"""
    path = group.loc[first_exit_index:last_exit_index]
    lows = pd.to_numeric(path["low"], errors="coerce").astype(float).to_numpy()
    highs = pd.to_numeric(path["high"], errors="coerce").astype(float).to_numpy()
    hit_stop = lows <= stop_price
    hit_target = highs >= target_price
    reasons = np.full(len(path), "", dtype=object)
    prices = np.full(len(path), np.nan)
    reasons[hit_stop] = "stop_loss"
    prices[hit_stop] = stop_price
    target_only = (reasons == "") & hit_target
    reasons[target_only] = "take_profit"
    prices[target_only] = target_price
    hit_positions = np.flatnonzero(reasons != "")
    if len(hit_positions) == 0:
        return last_exit_index, float(group.loc[last_exit_index, "close"]), "max_holding"
    first = int(hit_positions[0])
    return int(path.index[first]), float(prices[first]), str(reasons[first])


def run_single_strategy_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    *,
    timeframe: str = "",
) -> BacktestResult:
    orders = strategy.generate_orders(bars, timeframe=timeframe)
    strategy_filter_decisions = collect_strategy_filter_decisions([strategy])
    result = run_order_backtest(bars, orders, config or BacktestConfig())
    return _with_strategy_filter_decisions(result, strategy_filter_decisions)


def run_order_backtest(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    normalized = normalize_bars(bars)
    if orders.empty:
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        decisions = pd.DataFrame(columns=ORDER_DECISION_COLUMNS)
        equity = build_equity_curve(trades, cfg.initial_equity)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=_order_statistics(trades, equity, decisions),
            order_decisions=decisions,
        )

    trades: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    sorted_orders = _sorted_order_records(orders)
    seen_order_ids: set[str] = set()
    if normalized.empty:
        decisions = []
        for order in sorted_orders:
            reason = (
                order_duplicate_reject_reason(order, seen_order_ids)
                or order_preflight_reject_reason(order)
                or "no_bars"
            )
            decisions.append(order_decision_record(order, "rejected", reason))
        trades = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades, cfg.initial_equity)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=_order_statistics(trades, equity, pd.DataFrame(decisions, columns=ORDER_DECISION_COLUMNS)),
            order_decisions=pd.DataFrame(decisions, columns=ORDER_DECISION_COLUMNS),
        )

    bars_by_symbol = {
        symbol: group.reset_index(drop=True)
        for symbol, group in normalized.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False)
    }
    open_until: dict[str, int] = {}
    for order in sorted_orders:
        duplicate_rejection = order_duplicate_reject_reason(order, seen_order_ids)
        if duplicate_rejection:
            decisions.append(order_decision_record(order, "rejected", duplicate_rejection))
            continue
        preflight_rejection = order_preflight_reject_reason(order)
        if preflight_rejection:
            decisions.append(order_decision_record(order, "rejected", preflight_rejection))
            continue
        symbol = normalize_symbol(order["stock_code"])
        group = bars_by_symbol.get(symbol)
        if group is None or group.empty:
            decisions.append(order_decision_record(order, "rejected", "no_bars"))
            continue
        signal_index = int(order["signal_bar_index"])
        if signal_index < open_until.get(symbol, -1):
            decisions.append(order_decision_record(order, "rejected", "already_open"))
            continue
        execution = coerce_order_execution_result(
            simulate_order_trade_with_rejection(group, order, signal_index, cfg),
            order=order,
        )
        trade = execution.trade
        if trade is None:
            decisions.append(order_decision_record(order, "rejected", execution.reject_reason or "no_fill", execution=execution))
            continue
        trades.append(trade)
        decisions.append(
            order_decision_record(
                order,
                "accepted",
                "",
                trade=trade,
                execution=execution,
                capital_fraction=1.0,
                risk_fraction=_trade_risk_fraction(trade),
                margin_fraction=1.0,
            )
        )
        open_until[symbol] = int(trade["_exit_index"])

    decisions_df = pd.DataFrame(decisions, columns=ORDER_DECISION_COLUMNS)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades_df, cfg.initial_equity)
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity,
            stats=_order_statistics(trades_df, equity, decisions_df),
            order_decisions=decisions_df,
        )

    trades_df = _sort_trades_for_statistics(trades_df.drop(columns=["_exit_index"]))
    trades_df = trades_df[TRADE_COLUMNS]
    equity = build_equity_curve(trades_df, cfg.initial_equity)
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity,
        stats=_order_statistics(trades_df, equity, decisions_df),
        order_decisions=decisions_df,
    )


def _trade_statistics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, float]:
    """合并逐笔统计和净值曲线统计；净值指标以初始资金点为基准。"""
    stats = compute_trade_statistics(trades)
    stats.update(compute_equity_statistics(equity_curve))
    return stats


def _order_statistics(trades: pd.DataFrame, equity_curve: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, float]:
    stats = _trade_statistics(trades, equity_curve)
    stats.update(summarize_order_decisions(decisions))
    stats.update(summarize_strategy_filter_decisions(empty_strategy_filter_decisions()))
    return stats


def _with_strategy_filter_decisions(result: BacktestResult, filter_decisions: pd.DataFrame) -> BacktestResult:
    stats = dict(result.stats)
    stats.update(summarize_strategy_filter_decisions(filter_decisions))
    return BacktestResult(
        trades=result.trades,
        equity_curve=result.equity_curve,
        stats=stats,
        order_decisions=result.order_decisions,
        strategy_filter_decisions=filter_decisions,
    )


def _sort_orders_for_execution(orders: pd.DataFrame) -> pd.DataFrame:
    validate_order_frame_columns(orders)
    result = orders.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce")
    return result.sort_values(["signal_date", "stock_code", "order_id"]).reset_index(drop=True)


def _sorted_order_records(orders: pd.DataFrame) -> list[dict[str, object]]:
    """按撮合顺序输出订单记录，避免热路径反复创建 Series 行对象。"""
    return _sort_orders_for_execution(orders).to_dict("records")


def validate_order_frame_columns(orders: pd.DataFrame, *, extra_required: tuple[str, ...] = ()) -> None:
    """校验订单表结构；列缺失属于接入错误，必须用清晰异常暴露。"""
    missing = sorted(ORDER_REQUIRED_COLUMNS.union(extra_required).difference(orders.columns))
    if missing:
        raise ValueError(f"订单缺少必要字段：{', '.join(missing)}")


def order_duplicate_reject_reason(order: Mapping[str, object], seen_order_ids: set[str]) -> str:
    """同一轮回测内订单 ID 必须唯一；空 ID 交给原有撮合规则处理。"""
    order_id = _as_text(order.get("order_id", "")).strip()
    if not order_id:
        return ""
    if order_id in seen_order_ids:
        return "duplicate_order_id"
    seen_order_ids.add(order_id)
    return ""


def order_preflight_reject_reason(order: Mapping[str, object]) -> str:
    """订单进入撮合前的关键字段预检；失败必须进入决策日志。"""
    if not _as_text(order.get("order_id", "")).strip():
        return "invalid_order"
    if not _as_text(order.get("event_id", "")).strip():
        return "invalid_order"
    if not normalize_symbol(order.get("stock_code", "")):
        return "invalid_order"
    if str(order.get("side", "")).strip().lower() not in {"long", "short"}:
        return "invalid_order"
    if pd.isna(pd.to_datetime(order.get("signal_date", pd.NaT), errors="coerce")):
        return "invalid_order"
    try:
        signal_index = int(order.get("signal_bar_index", -1))
    except (TypeError, ValueError):
        return "invalid_order"
    if signal_index < 0:
        return "invalid_order"
    if not all(_is_positive_number(order.get(column, None)) for column in ("entry_price", "stop_price", "target_price")):
        return "invalid_order"
    if _as_float(order.get("entry_price", 0.0)) == _as_float(order.get("stop_price", 0.0)):
        return "invalid_order"
    max_holding = order.get("max_holding_bars", 1)
    if not pd.isna(max_holding) and _as_int(max_holding, default=0) < 1:
        return "invalid_order"
    return ""


def order_decision_record(
    order: pd.Series,
    status: str,
    reason: str,
    *,
    trade: dict[str, object] | None = None,
    capital_fraction: float = 0.0,
    risk_fraction: float = 0.0,
    margin_fraction: float = 0.0,
    sector: str = "",
    portfolio_priority: int | None = None,
    execution: OrderExecutionResult | None = None,
) -> dict[str, object]:
    """生成订单决策日志；记录未成交、被拒绝和接受的统一原因。"""
    source = trade or order
    execution_metrics = _decision_execution_metrics(order, trade, execution)
    return {
        "order_id": _field(source, order, "order_id", ""),
        "event_id": _field(source, order, "event_id", ""),
        "event_type": _as_text(_field(source, order, "event_type", "")),
        "strategy_name": _field(source, order, "strategy_name", ""),
        "detector_name": _field(source, order, "detector_name", ""),
        "stock_code": _field(source, order, "stock_code", ""),
        "timeframe": _field(source, order, "timeframe", ""),
        "signal_date": _field(order, source, "signal_date", pd.NaT),
        "signal_bar_index": _as_int(_field(order, source, "signal_bar_index", -1), default=-1),
        "side": _field(source, order, "side", ""),
        "planned_entry_price": _as_float(_field(source, order, "planned_entry_price", _field(order, source, "entry_price", 0.0))),
        "entry_date": _field(source, order, "entry_date", pd.NaT) if trade is not None else pd.NaT,
        **execution_metrics,
        "status": status,
        "reason": reason,
        "portfolio_priority": _as_int(
            portfolio_priority if portfolio_priority is not None else _field(order, source, "_portfolio_priority", 0),
            default=0,
        ),
        "capital_fraction": float(capital_fraction),
        "risk_fraction": float(risk_fraction),
        "margin_fraction": float(margin_fraction),
        "sector": str(sector),
    }


def _decision_execution_metrics(
    order: pd.Series,
    trade: dict[str, object] | None,
    execution: OrderExecutionResult | None,
) -> dict[str, float]:
    if execution is not None:
        return {
            "actual_entry_price": float(execution.actual_entry_price),
            "actual_risk_pct": float(execution.actual_risk_pct),
            "actual_chase_pct": float(execution.actual_chase_pct),
            "actual_reward_to_risk": float(execution.actual_reward_to_risk),
        }
    if trade is None:
        return {
            "actual_entry_price": 0.0,
            "actual_risk_pct": 0.0,
            "actual_chase_pct": 0.0,
            "actual_reward_to_risk": 0.0,
        }
    return compute_order_execution_metrics(
        order,
        str(_field(trade, order, "side", "")),
        _as_float(_field(trade, order, "entry_price", 0.0)),
        _as_float(_field(trade, order, "stop_price", 0.0)),
        _as_float(_field(trade, order, "target_price", 0.0)),
    )


def _field(primary: object, fallback: object, key: str, default: object) -> object:
    if isinstance(primary, dict) and key in primary:
        return primary[key]
    if hasattr(primary, "get"):
        value = primary.get(key, default)
        if value is not default:
            return value
    if isinstance(fallback, dict) and key in fallback:
        return fallback[key]
    if hasattr(fallback, "get"):
        return fallback.get(key, default)
    return default


def _as_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value)


def _is_positive_number(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _as_int(value: object, *, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _trade_risk_fraction(trade: dict[str, object]) -> float:
    entry_price = float(trade["entry_price"])
    risk_per_share = float(trade["risk_per_share"])
    if entry_price <= 0 or risk_per_share <= 0:
        return 0.0
    return float(risk_per_share / entry_price)


def _sort_trades_for_statistics(trades: pd.DataFrame) -> pd.DataFrame:
    """成交按真实入场时间排序，避免统计结果受股票代码顺序影响。"""
    if trades.empty or "entry_date" not in trades.columns:
        return trades
    result = trades.copy()
    result["entry_date"] = pd.to_datetime(result["entry_date"], errors="coerce")
    return result.sort_values(["entry_date", "stock_code", "order_id"]).reset_index(drop=True)
