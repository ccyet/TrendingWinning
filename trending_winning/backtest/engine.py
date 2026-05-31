from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from trending_winning.backtest.execution import (
    apply_slippage,
    liquid_bar_mask,
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
from trending_winning.backtest.order_backtest import (
    market_bar_count as _market_bar_count,
    order_statistics as _order_statistics,
    run_order_backtest as run_order_backtest,
    run_order_backtest_from_normalized as run_order_backtest_from_normalized,
    sort_trades_for_statistics as _sort_trades_for_statistics,
    trade_risk_fraction as _trade_risk_fraction,
)
from trending_winning.backtest.portfolio_equity import build_single_position_equity_curve_from_normalized
from trending_winning.backtest.order_decisions import (
    order_decision_record as order_decision_record,
    order_duplicate_reject_reason as order_duplicate_reject_reason,
    order_preflight_reject_reason as order_preflight_reject_reason,
    validate_order_frame_columns as validate_order_frame_columns,
)
from trending_winning.backtest.position_gate import apply_single_position_gate
from trending_winning.backtest.stats import (
    build_equity_curve,
    summarize_strategy_filter_decisions,
)
from trending_winning.backtest.trailing_take_profit import (
    trailing_take_profit_enabled as is_trailing_take_profit_enabled,
    trailing_take_profit_masks as compute_trailing_take_profit_masks,
)
from trending_winning.data.schema import normalize_bars
from trending_winning.strategies.base import Strategy
from trending_winning.strategies.runtime import execute_strategy


def run_backtest(scanned_bars: pd.DataFrame, config: BacktestConfig | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    if cfg.take_profit_pct <= 0 or cfg.stop_loss_pct <= 0:
        raise ValueError("旧突破回测的 take_profit_pct 和 stop_loss_pct 必须大于 0。")

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
    equity = build_single_position_equity_curve_from_normalized(normalize_bars(scanned_bars), trades_df, cfg.initial_equity)
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


def _as_float(value: object) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
