from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from trending_winning.backtest.execution import (
    OrderExecutionResult,
    apply_slippage,
    coerce_order_execution_result,
    compute_order_execution_metrics,
    is_favorable_target,
    is_protective_stop,
    liquid_bar_mask,
    normalize_order_side,
    simulate_order_trade_with_rejection,
    trade_path_metrics,
    validate_backtest_config,
)
from trending_winning.backtest.indicators import completed_bar_moving_average
from trending_winning.backtest.models import (
    BacktestConfig as BacktestConfig,
    BacktestResult as BacktestResult,
    ORDER_DECISION_COLUMNS as ORDER_DECISION_COLUMNS,
    ORDER_REQUIRED_COLUMNS as ORDER_REQUIRED_COLUMNS,
    TRADE_COLUMNS as TRADE_COLUMNS,
)
from trending_winning.backtest.position_gate import apply_single_position_gate
from trending_winning.backtest.stats import (
    build_equity_curve,
    compute_equity_statistics,
    compute_trade_statistics,
    summarize_exit_reasons,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.backtest.trailing_take_profit import (
    trailing_take_profit_enabled as is_trailing_take_profit_enabled,
    trailing_take_profit_masks as compute_trailing_take_profit_masks,
)
from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.strategies.base import Strategy
from trending_winning.strategies.diagnostics import empty_strategy_filter_decisions
from trending_winning.strategies.runtime import execute_strategy


def run_backtest(scanned_bars: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)

    trigger_records = _legacy_trigger_records(scanned_bars)
    candidates: list[dict[str, object]] = []
    decisions: list[tuple[int, dict[str, object]]] = []
    for order_sequence, trigger in enumerate(trigger_records):
        group = trigger["group"]
        index = int(trigger["bar_index"])
        symbol = str(trigger["stock_code"])
        trade = _simulate_trade(group, index, symbol, cfg)
        if trade is None:
            decisions.append(
                (
                    order_sequence,
                    _legacy_signal_rejection_record(
                        group.loc[index],
                        index,
                        symbol,
                        _legacy_signal_reject_reason(group, index),
                    ),
                )
            )
            continue
        candidates.append({"trade": trade, "order_sequence": order_sequence})

    trades: list[dict[str, object]] = []
    # 旧版突破回测也按单策略满仓处理：任意股票有持仓时，全局不再开第二笔。
    for gate_decision in apply_single_position_gate(candidates):
        candidate = gate_decision.candidate
        trade = candidate["trade"]
        if not isinstance(trade, Mapping):
            continue
        order_sequence = int(candidate.get("order_sequence", 0))
        trade_record = dict(trade)
        if gate_decision.status == "rejected":
            decisions.append((order_sequence, _legacy_order_decision_record(trade_record, "rejected", gate_decision.reason)))
            continue
        trades.append(trade_record)
        decisions.append((order_sequence, _legacy_order_decision_record(trade_record, "accepted", "")))

    decision_records = [record for _, record in sorted(decisions, key=lambda item: item[0])]
    decisions_df = pd.DataFrame(decision_records, columns=ORDER_DECISION_COLUMNS)
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=TRADE_COLUMNS)
        equity = build_equity_curve(trades_df, cfg.initial_equity)
        return BacktestResult(
            trades=trades_df,
            equity_curve=equity,
            stats=_order_statistics(
                trades_df,
                equity,
                decisions_df,
                market_bar_count=_market_bar_count(scanned_bars),
            ),
            order_decisions=decisions_df,
        )

    trades_df = _sort_trades_for_statistics(trades_df.drop(columns=["_exit_index"]))
    trades_df = trades_df[TRADE_COLUMNS]
    equity = build_equity_curve(trades_df, cfg.initial_equity)
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity,
        stats=_order_statistics(
            trades_df,
            equity,
            decisions_df,
            market_bar_count=_market_bar_count(scanned_bars),
        ),
        order_decisions=decisions_df,
    )


def _legacy_order_decision_record(trade: dict[str, object], status: str, reason: str) -> dict[str, object]:
    """旧版突破信号也落到统一决策日志，便于统计被满仓门控过滤的触发。"""
    capital_fraction = 1.0 if status == "accepted" else 0.0
    return order_decision_record(
        trade,
        status,
        reason,
        trade=trade,
        capital_fraction=capital_fraction,
        risk_fraction=_trade_risk_fraction(trade) if status == "accepted" else 0.0,
        margin_fraction=capital_fraction,
    )


def _legacy_trigger_records(scanned_bars: pd.DataFrame) -> list[dict[str, object]]:
    """提取旧版突破触发并按全市场信号时间排序，保证复盘日志跨股票连续。"""
    records: list[dict[str, object]] = []
    for symbol, group in scanned_bars.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False):
        group = group.reset_index(drop=True)
        for index, triggered in enumerate(group["breakout_trigger"].tolist()):
            if not bool(triggered):
                continue
            records.append(
                {
                    "signal_date": pd.to_datetime(group.loc[index, "date"], errors="coerce"),
                    "stock_code": str(symbol),
                    "bar_index": int(index),
                    "group": group,
                }
            )
    return sorted(
        records,
        key=lambda record: (
            bool(pd.isna(record["signal_date"])),
            pd.Timestamp.max if pd.isna(record["signal_date"]) else pd.Timestamp(record["signal_date"]),
            str(record["stock_code"]),
            int(record["bar_index"]),
        ),
    )


def _legacy_signal_rejection_record(row: pd.Series, signal_index: int, symbol: str, reason: str) -> dict[str, object]:
    """记录没有形成候选成交的旧版触发，避免最后一根 K 或坏价格被静默丢弃。"""
    order = _legacy_signal_order_record(row, signal_index, symbol)
    return order_decision_record(order, "rejected", reason)


def _legacy_signal_order_record(row: pd.Series, signal_index: int, symbol: str) -> dict[str, object]:
    signal_date = row.get("date", pd.NaT)
    planned_entry_price = _legacy_planned_entry_price(row)
    event_id = f"legacy_scan:{symbol}:{pd.Timestamp(signal_date).isoformat()}"
    return {
        "order_id": f"legacy_breakout:{symbol}:{pd.Timestamp(signal_date).isoformat()}",
        "event_id": event_id,
        "event_type": "legacy_breakout",
        "strategy_name": "legacy_breakout",
        "detector_name": "legacy_scan",
        "stock_code": symbol,
        "timeframe": "",
        "signal_date": signal_date,
        "signal_bar_index": int(signal_index),
        "side": "long",
        "entry_price": planned_entry_price,
        "stop_price": 0.0,
        "target_price": 0.0,
        "metadata": {},
    }


def _legacy_signal_reject_reason(group: pd.DataFrame, signal_index: int) -> str:
    if signal_index >= len(group) - 1:
        return "no_fill"
    row = group.loc[signal_index]
    planned_entry_price = _legacy_planned_entry_price(row)
    return "invalid_order" if planned_entry_price <= 0 else "no_fill"


def _simulate_trade(
    group: pd.DataFrame,
    entry_index: int,
    symbol: str,
    cfg: BacktestConfig,
) -> dict[str, object] | None:
    entry = group.loc[entry_index]
    planned_entry_price = _legacy_planned_entry_price(entry)
    if planned_entry_price <= 0:
        return None
    entry_price = apply_slippage(planned_entry_price, 1.0, cfg)
    target_price = entry_price * (1.0 + cfg.take_profit_pct)
    stop_price = entry_price * (1.0 - cfg.stop_loss_pct)
    last_exit_index = min(len(group) - 1, entry_index + cfg.max_holding_bars)
    if last_exit_index <= entry_index:
        return None

    exit_index, raw_exit_price, exit_reason = _first_legacy_long_exit(
        group,
        first_exit_index=entry_index + 1,
        last_exit_index=last_exit_index,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        cfg=cfg,
    )
    exit_price = apply_slippage(raw_exit_price, -1.0, cfg)

    return_pct = ((exit_price / entry_price - 1.0) - 2.0 * cfg.fee_rate) * 100.0
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
        "planned_entry_price": planned_entry_price,
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


def _legacy_planned_entry_price(row: pd.Series) -> float:
    trigger_price = row.get("trigger_price", pd.NA)
    source_price = trigger_price if pd.notna(trigger_price) else row.get("close", 0.0)
    return _as_float(source_price)


def _first_legacy_long_exit(
    group: pd.DataFrame,
    *,
    first_exit_index: int,
    last_exit_index: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    cfg: BacktestConfig,
) -> tuple[int, float, str]:
    """旧版突破回测的长仓退出；保持 stop 优先于 target 的原始语义。"""
    path = group.loc[first_exit_index:last_exit_index]
    opens = pd.to_numeric(path["open"], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(path["low"], errors="coerce").astype(float).to_numpy()
    highs = pd.to_numeric(path["high"], errors="coerce").astype(float).to_numpy()
    liquid = liquid_bar_mask(path)
    gap_stop = liquid & (opens <= stop_price)
    gap_target = liquid & (opens >= target_price)
    hit_stop = liquid & (lows <= stop_price)
    hit_target = liquid & (highs >= target_price)
    hit_trailing, trailing_prices = _legacy_long_trailing_take_profit(
        path,
        entry_price,
        cfg,
        moving_average=completed_bar_moving_average(group, path.index, cfg.trailing_take_profit_ma_period),
    )
    gap_trailing = hit_trailing & (opens <= trailing_prices)
    reasons = np.full(len(path), "", dtype=object)
    prices = np.full(len(path), np.nan)

    reasons[gap_stop] = "stop_loss"
    prices[gap_stop] = opens[gap_stop]
    gap_target_only = (reasons == "") & gap_target
    reasons[gap_target_only] = "take_profit"
    prices[gap_target_only] = opens[gap_target_only]
    gap_trailing_only = (reasons == "") & gap_trailing
    reasons[gap_trailing_only] = "trailing_take_profit"
    prices[gap_trailing_only] = opens[gap_trailing_only]

    stop_only = (reasons == "") & hit_stop
    reasons[stop_only] = "stop_loss"
    prices[stop_only] = stop_price
    target_trailing_conflict = (reasons == "") & hit_target & hit_trailing
    reasons[target_trailing_conflict] = "trailing_take_profit"
    prices[target_trailing_conflict] = trailing_prices[target_trailing_conflict]
    target_only = (reasons == "") & hit_target
    reasons[target_only] = "take_profit"
    prices[target_only] = target_price
    trailing_only = (reasons == "") & hit_trailing
    reasons[trailing_only] = "trailing_take_profit"
    prices[trailing_only] = trailing_prices[trailing_only]
    hit_positions = np.flatnonzero(reasons != "")
    if len(hit_positions) == 0:
        return last_exit_index, float(group.loc[last_exit_index, "close"]), "max_holding"
    first = int(hit_positions[0])
    return int(path.index[first]), float(prices[first]), str(reasons[first])


def _legacy_long_trailing_take_profit(
    path: pd.DataFrame,
    entry_price: float,
    cfg: BacktestConfig,
    *,
    moving_average: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """旧版长仓复用回撤止盈口径；回撤线只使用上一根完成 K 的信息。"""
    if not _trailing_take_profit_enabled(cfg) or path.empty or entry_price <= 0:
        return np.full(len(path), False), np.full(len(path), np.nan)
    highs = pd.to_numeric(path["high"], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(path["low"], errors="coerce").astype(float).to_numpy()
    opens = pd.to_numeric(path["open"], errors="coerce").astype(float).to_numpy()
    liquid = liquid_bar_mask(path)
    result = compute_trailing_take_profit_masks(
        opens,
        highs,
        lows,
        liquid,
        side="long",
        entry_price=entry_price,
        activation_pct=float(cfg.trailing_take_profit_activation_pct),
        drawdown_pct=float(cfg.trailing_take_profit_drawdown_pct),
        moving_average=moving_average,
    )
    return result.hit, result.prices


def _trailing_take_profit_enabled(cfg: BacktestConfig) -> bool:
    return is_trailing_take_profit_enabled(
        float(cfg.trailing_take_profit_activation_pct),
        float(cfg.trailing_take_profit_drawdown_pct),
        int(cfg.trailing_take_profit_ma_period),
    )


def run_single_strategy_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    *,
    timeframe: str = "",
) -> BacktestResult:
    return run_single_strategy_backtest_from_normalized(
        normalize_bars(bars),
        strategy,
        config or BacktestConfig(),
        timeframe=timeframe,
    )


def run_single_strategy_backtest_from_normalized(
    normalized_bars: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    *,
    timeframe: str = "",
) -> BacktestResult:
    """基于已标准化 K 线执行单策略回测；实验和 sweep 热路径复用同一批 K。"""
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    strategy_run = execute_strategy(strategy, normalized_bars, timeframe=timeframe)
    result = run_order_backtest_from_normalized(normalized_bars, strategy_run.orders, cfg)
    return _with_strategy_filter_decisions(result, strategy_run.filter_decisions)


def run_order_backtest(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return _run_order_backtest_from_normalized(normalize_bars(bars), orders, cfg)


def run_order_backtest_from_normalized(
    normalized_bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """基于已标准化 K 线做订单回测；参数遍历热路径用它避免重复 normalize。"""
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return _run_order_backtest_from_normalized(normalized_bars, orders, cfg)


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
            stats=_order_statistics(
                trades,
                equity,
                decisions,
                market_bar_count=_market_bar_count(normalized),
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
            stats=_order_statistics(
                trades,
                equity,
                pd.DataFrame(empty_decisions, columns=ORDER_DECISION_COLUMNS),
                market_bar_count=_market_bar_count(normalized),
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
                    risk_fraction=_trade_risk_fraction(trade),
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
            stats=_order_statistics(
                trades_df,
                equity,
                decisions_df,
                market_bar_count=_market_bar_count(normalized),
            ),
            order_decisions=decisions_df,
        )

    trades_df = _sort_trades_for_statistics(trades_df.drop(columns=["_exit_index"]))
    trades_df = trades_df[TRADE_COLUMNS]
    equity = build_equity_curve(trades_df, cfg.initial_equity)
    return BacktestResult(
        trades=trades_df,
        equity_curve=equity,
        stats=_order_statistics(
            trades_df,
            equity,
            decisions_df,
            market_bar_count=_market_bar_count(normalized),
        ),
        order_decisions=decisions_df,
    )


def _trade_statistics(trades: pd.DataFrame, equity_curve: pd.DataFrame) -> dict[str, object]:
    """合并逐笔统计和净值曲线统计；净值指标以初始资金点为基准。"""
    stats = compute_trade_statistics(trades)
    stats.update(summarize_exit_reasons(trades))
    stats.update(compute_equity_statistics(equity_curve))
    return stats


def _order_statistics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    market_bar_count: int = 0,
) -> dict[str, object]:
    stats = _trade_statistics(trades, equity_curve)
    stats.update(_single_position_exposure_statistics(stats, market_bar_count=market_bar_count))
    stats.update(summarize_order_decisions(decisions))
    stats.update(summarize_strategy_filter_decisions(empty_strategy_filter_decisions()))
    return stats


def _single_position_exposure_statistics(stats: Mapping[str, object], *, market_bar_count: int) -> dict[str, float]:
    """单策略是满仓进出，场内比例应按全市场时间轴计算，不能沿用逐 K 组合净值默认值。"""
    market_count = max(int(market_bar_count), 0)
    exposure_bars = max(_as_float(stats.get("exposure_bars", 0.0)), 0.0)
    ratio = min(1.0, exposure_bars / market_count) if market_count > 0 else 0.0
    return {
        "market_bar_count": float(market_count),
        "exposure_bar_ratio": float(ratio),
    }


def _market_bar_count(bars: pd.DataFrame) -> int:
    if bars.empty or "date" not in bars.columns:
        return 0
    dates = pd.to_datetime(bars["date"], errors="coerce")
    return int(dates.dropna().nunique())


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
    side = normalize_order_side(order.get("side", ""))
    if side not in {"long", "short"}:
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
    entry_price = _as_float(order.get("entry_price", 0.0))
    stop_price = _as_float(order.get("stop_price", 0.0))
    target_price = _as_float(order.get("target_price", 0.0))
    if not is_protective_stop(side, entry_price, stop_price):
        return "invalid_order"
    if not is_favorable_target(side, entry_price, target_price):
        return "target_not_favorable"
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
    side = _decision_side(_field(source, order, "side", ""))
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
        "side": side,
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


def _decision_side(value: object) -> str:
    normalized = normalize_order_side(value)
    return normalized or _as_text(value)


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
