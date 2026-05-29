from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from trending_winning.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    ORDER_DECISION_COLUMNS,
    TRADE_COLUMNS,
    order_decision_record,
    order_duplicate_reject_reason,
    order_preflight_reject_reason,
    validate_order_frame_columns,
)
from trending_winning.backtest.execution import (
    OrderExecutionResult,
    coerce_order_execution_result,
    simulate_order_trade_with_rejection,
    validate_backtest_config,
)
from trending_winning.backtest.stats import (
    compute_equity_statistics,
    compute_trade_statistics,
    summarize_order_decisions,
    summarize_strategy_filter_decisions,
)
from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.strategies.base import ORDER_COLUMNS, Strategy
from trending_winning.strategies.diagnostics import collect_strategy_filter_decisions, empty_strategy_filter_decisions

PORTFOLIO_COLUMNS = TRADE_COLUMNS + [
    "raw_return_pct",
    "capital_fraction",
    "margin_fraction",
    "risk_fraction",
    "sector",
    "portfolio_priority",
]


@dataclass(frozen=True)
class PortfolioConfig:
    """组合回测参数；capital_fraction 是名义仓位，margin_fraction 是组合资金占用。"""

    max_open_positions: int = 5
    capital_per_trade: float | None = None
    risk_per_trade: float | None = None
    max_capital_per_trade: float = 1.0
    short_margin_rate: float = 1.0
    reserve_cash: float = 0.0
    allow_same_symbol_overlap: bool = False
    strategy_priority: Mapping[str, int] = field(default_factory=dict)
    strategy_capital_limit: Mapping[str, float] = field(default_factory=dict)
    sector_capital_limit: Mapping[str, float] = field(default_factory=dict)
    symbol_sector_map: Mapping[str, str] = field(default_factory=dict)
    sector_metadata_key: str = "sector"
    default_sector: str = "UNKNOWN"


@dataclass(frozen=True)
class PortfolioCandidateSet:
    """已完成单笔撮合的候选成交；组合参数遍历可复用，避免重复模拟成交路径。"""

    candidates: tuple[dict[str, object], ...] = field(default_factory=tuple)
    rejections: tuple[dict[str, object], ...] = field(default_factory=tuple)


def run_portfolio_backtest(
    bars: pd.DataFrame,
    strategies: Sequence[Strategy],
    config: BacktestConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    *,
    timeframe: str = "",
) -> BacktestResult:
    cfg = config or BacktestConfig()
    pcfg = portfolio_config or PortfolioConfig()
    validate_backtest_config(cfg)
    _validate_portfolio_config(pcfg)

    normalized = normalize_bars(bars)
    orders = _collect_strategy_orders(normalized, strategies, timeframe=timeframe)
    strategy_filter_decisions = collect_strategy_filter_decisions(strategies)
    return _run_portfolio_orders(normalized, orders, cfg, pcfg, strategy_filter_decisions=strategy_filter_decisions)


def collect_strategy_orders(
    bars: pd.DataFrame,
    strategies: Sequence[Strategy],
    *,
    timeframe: str = "",
) -> pd.DataFrame:
    """只生成策略订单；参数遍历可复用订单流，避免重复执行 detector。"""
    return _collect_strategy_orders(normalize_bars(bars), strategies, timeframe=timeframe)


def collect_strategy_orders_from_normalized(
    normalized_bars: pd.DataFrame,
    strategies: Sequence[Strategy],
    *,
    timeframe: str = "",
) -> pd.DataFrame:
    """基于已标准化 K 线生成策略订单；高性能遍历用它避免重复 normalize。"""
    return _collect_strategy_orders(normalized_bars, strategies, timeframe=timeframe)


def run_portfolio_order_backtest(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> BacktestResult:
    """基于已生成订单做组合撮合；用于高性能参数遍历和外部策略接入。"""
    cfg = config or BacktestConfig()
    pcfg = portfolio_config or PortfolioConfig()
    validate_backtest_config(cfg)
    _validate_portfolio_config(pcfg)
    return _run_portfolio_orders(normalize_bars(bars), orders, cfg, pcfg)


def prepare_portfolio_candidates(
    bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> PortfolioCandidateSet:
    """将订单撮合为候选成交；不做仓位分配，便于组合层参数复用。"""
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return _prepare_portfolio_candidates_from_normalized(normalize_bars(bars), orders, cfg)


def prepare_portfolio_candidates_from_normalized(
    normalized_bars: pd.DataFrame,
    orders: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> PortfolioCandidateSet:
    """基于已标准化 K 线生成候选成交；供参数遍历复用。"""
    cfg = config or BacktestConfig()
    validate_backtest_config(cfg)
    return _prepare_portfolio_candidates_from_normalized(normalized_bars, orders, cfg)


def run_portfolio_candidate_backtest(
    bars: pd.DataFrame,
    candidate_set: PortfolioCandidateSet,
    config: BacktestConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> BacktestResult:
    """基于候选成交做组合分配；不会重新执行 detector 或单笔撮合。"""
    cfg = config or BacktestConfig()
    pcfg = portfolio_config or PortfolioConfig()
    validate_backtest_config(cfg)
    _validate_portfolio_config(pcfg)
    return _run_portfolio_candidates(normalize_bars(bars), candidate_set, cfg, pcfg)


def run_portfolio_candidate_backtest_from_normalized(
    normalized_bars: pd.DataFrame,
    candidate_set: PortfolioCandidateSet,
    config: BacktestConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> BacktestResult:
    """基于已标准化 K 线做候选组合分配；不会重新标准化行情。"""
    cfg = config or BacktestConfig()
    pcfg = portfolio_config or PortfolioConfig()
    validate_backtest_config(cfg)
    _validate_portfolio_config(pcfg)
    return _run_portfolio_candidates(normalized_bars, candidate_set, cfg, pcfg)


def _run_portfolio_orders(
    normalized: pd.DataFrame,
    orders: pd.DataFrame,
    cfg: BacktestConfig,
    pcfg: PortfolioConfig,
    *,
    strategy_filter_decisions: pd.DataFrame | None = None,
) -> BacktestResult:
    candidate_set = _prepare_portfolio_candidates_from_normalized(normalized, orders, cfg)
    return _run_portfolio_candidates(
        normalized,
        candidate_set,
        cfg,
        pcfg,
        strategy_filter_decisions=strategy_filter_decisions,
    )


def _run_portfolio_candidates(
    normalized: pd.DataFrame,
    candidate_set: PortfolioCandidateSet,
    cfg: BacktestConfig,
    pcfg: PortfolioConfig,
    *,
    strategy_filter_decisions: pd.DataFrame | None = None,
) -> BacktestResult:
    filter_decisions = strategy_filter_decisions if strategy_filter_decisions is not None else empty_strategy_filter_decisions()
    if not candidate_set.candidates and not candidate_set.rejections:
        trades = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
        decisions = pd.DataFrame(columns=ORDER_DECISION_COLUMNS)
        equity = _build_portfolio_equity_curve_from_normalized(normalized, trades, cfg.initial_equity)
        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            stats=_portfolio_statistics(trades, equity, decisions, filter_decisions),
            order_decisions=decisions,
            strategy_filter_decisions=filter_decisions,
        )

    trades, decisions = _simulate_allocated_candidates(candidate_set, pcfg)
    if trades.empty:
        trades = pd.DataFrame(columns=PORTFOLIO_COLUMNS)
    equity = _build_portfolio_equity_curve_from_normalized(normalized, trades, cfg.initial_equity)
    return BacktestResult(
        trades=trades,
        equity_curve=equity,
        stats=_portfolio_statistics(trades, equity, decisions, filter_decisions),
        order_decisions=decisions,
        strategy_filter_decisions=filter_decisions,
    )


def _portfolio_statistics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    decisions: pd.DataFrame,
    filter_decisions: pd.DataFrame,
) -> dict[str, float]:
    stats = compute_trade_statistics(trades)
    stats.update(compute_equity_statistics(equity_curve))
    stats.update(summarize_order_decisions(decisions))
    stats.update(summarize_strategy_filter_decisions(filter_decisions))
    return stats


def _collect_strategy_orders(bars: pd.DataFrame, strategies: Sequence[Strategy], *, timeframe: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy in strategies:
        orders = strategy.generate_orders(bars, timeframe=timeframe)
        if orders.empty:
            continue
        frame = orders.copy()
        frame["strategy_name"] = frame["strategy_name"].replace("", strategy.name).fillna(strategy.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _prepare_portfolio_candidates_from_normalized(
    normalized: pd.DataFrame,
    orders: pd.DataFrame,
    cfg: BacktestConfig,
) -> PortfolioCandidateSet:
    if orders.empty:
        return PortfolioCandidateSet()
    if normalized.empty:
        return PortfolioCandidateSet(rejections=tuple(_no_bar_order_rejections(orders)))
    bars_by_symbol = {
        symbol: group.reset_index(drop=True)
        for symbol, group in normalized.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False)
    }
    candidates, rejections = _candidate_trades_by_entry_time(bars_by_symbol, orders, cfg)
    return PortfolioCandidateSet(candidates=tuple(candidates), rejections=tuple(rejections))


def _no_bar_order_rejections(orders: pd.DataFrame) -> list[dict[str, object]]:
    """行情为空时仍逐笔记录订单拒绝原因，避免组合统计静默丢单。"""
    rejections: list[dict[str, object]] = []
    seen_order_ids: set[str] = set()
    for order in _sort_orders_for_simulation(orders).to_dict("records"):
        reason = order_duplicate_reject_reason(order, seen_order_ids) or order_preflight_reject_reason(order) or "no_bars"
        rejections.append({"order": order.copy(), "reason": reason})
    return rejections


def _simulate_allocated_candidates(
    candidate_set: PortfolioCandidateSet,
    pcfg: PortfolioConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _sort_candidates_for_portfolio(candidate_set.candidates, pcfg)
    decisions = [_rejection_decision(rejection, pcfg) for rejection in candidate_set.rejections]
    open_positions: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    for candidate in candidates:
        order = candidate["order"]
        trade = dict(candidate["trade"])
        entry_date = pd.Timestamp(trade["entry_date"])
        open_positions = [item for item in open_positions if pd.Timestamp(item["exit_date"]) > entry_date]
        symbol = str(trade["stock_code"])
        if _has_symbol_overlap(open_positions, symbol, pcfg):
            decisions.append(_order_decision(candidate, "rejected", "same_symbol_overlap"))
            continue
        if len(open_positions) >= pcfg.max_open_positions:
            decisions.append(_order_decision(candidate, "rejected", "max_open_positions"))
            continue
        sector = str(candidate["sector"])
        risk_fraction = float(candidate["risk_fraction"])
        capital_fraction = _next_capital_fraction(open_positions, order, sector, risk_fraction, pcfg)
        if capital_fraction <= 0:
            decisions.append(_order_decision(candidate, "rejected", "no_capital"))
            continue
        margin_fraction = _order_margin_fraction(order, capital_fraction, pcfg)
        raw_return_pct = float(trade["return_pct"])
        trade["raw_return_pct"] = raw_return_pct
        trade["capital_fraction"] = capital_fraction
        trade["margin_fraction"] = margin_fraction
        trade["risk_fraction"] = risk_fraction
        trade["sector"] = sector
        trade["portfolio_priority"] = int(order["_portfolio_priority"])
        trade["return_pct"] = raw_return_pct * capital_fraction
        trades.append(trade)
        decisions.append(_order_decision(candidate, "accepted", "", capital_fraction=capital_fraction, margin_fraction=margin_fraction))
        open_positions.append(
            {
                "stock_code": symbol,
                "strategy_name": str(order["strategy_name"]),
                "exit_date": trade["exit_date"],
                "capital_fraction": capital_fraction,
                "margin_fraction": margin_fraction,
                "sector": sector,
                "risk_fraction": risk_fraction,
            }
        )
    decisions_frame = _portfolio_order_decisions_frame(decisions)
    if not trades:
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS), decisions_frame
    frame = pd.DataFrame(trades).drop(columns=["_exit_index"])
    return frame[PORTFOLIO_COLUMNS], decisions_frame


def _candidate_trades_by_entry_time(
    bars_by_symbol: dict[str, pd.DataFrame],
    orders: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    seen_order_ids: set[str] = set()
    for order in _sort_orders_for_simulation(orders).to_dict("records"):
        duplicate_rejection = order_duplicate_reject_reason(order, seen_order_ids)
        if duplicate_rejection:
            rejections.append({"order": order.copy(), "reason": duplicate_rejection})
            continue
        preflight_rejection = order_preflight_reject_reason(order)
        if preflight_rejection:
            rejections.append({"order": order.copy(), "reason": preflight_rejection})
            continue
        symbol = normalize_symbol(order["stock_code"])
        group = bars_by_symbol.get(symbol)
        if group is None or group.empty:
            rejections.append({"order": order.copy(), "reason": "no_bars"})
            continue
        execution = coerce_order_execution_result(
            simulate_order_trade_with_rejection(group, order, int(order["signal_bar_index"]), cfg),
            order=order,
        )
        trade = execution.trade
        if trade is None:
            rejections.append({"order": order.copy(), "reason": execution.reject_reason or "no_fill", "execution": execution})
            continue
        candidates.append(
            {
                "order": order.copy(),
                "trade": trade,
                "execution": execution,
                "risk_fraction": _trade_risk_fraction(trade),
            }
        )
    return candidates, rejections


def _sort_candidates_for_portfolio(
    candidates: Sequence[dict[str, object]],
    pcfg: PortfolioConfig,
) -> list[dict[str, object]]:
    enriched = [_candidate_with_portfolio_context(candidate, pcfg) for candidate in candidates]
    return sorted(
        enriched,
        key=lambda item: (
            pd.Timestamp(item["trade"]["entry_date"]),
            int(item["order"]["_portfolio_priority"]),
            str(item["trade"]["stock_code"]),
            str(item["trade"]["order_id"]),
        ),
    )


def _candidate_with_portfolio_context(candidate: dict[str, object], pcfg: PortfolioConfig) -> dict[str, object]:
    order = _order_with_portfolio_priority(candidate["order"], pcfg)
    return {
        "order": order,
        "trade": dict(candidate["trade"]),
        "execution": candidate.get("execution"),
        "sector": _order_sector(order, pcfg),
        "risk_fraction": float(candidate["risk_fraction"]),
    }


def _rejection_decision(rejection: dict[str, object], pcfg: PortfolioConfig) -> dict[str, object]:
    order = _order_with_portfolio_priority(rejection["order"], pcfg)
    sector = _order_sector(order, pcfg)
    record = order_decision_record(
        order,
        "rejected",
        str(rejection["reason"]),
        sector=sector,
        portfolio_priority=int(order["_portfolio_priority"]),
        execution=_execution_or_none(rejection.get("execution")),
    )
    record["_order_sequence"] = int(order.get("_order_sequence", 0))
    return record


def _order_decision(
    candidate: dict[str, object],
    status: str,
    reason: str,
    *,
    capital_fraction: float = 0.0,
    margin_fraction: float = 0.0,
) -> dict[str, object]:
    order = candidate["order"]
    trade = candidate["trade"]
    record = order_decision_record(
        order,
        status,
        reason,
        trade=trade,
        capital_fraction=capital_fraction,
        risk_fraction=float(candidate.get("risk_fraction", 0.0)),
        margin_fraction=margin_fraction,
        sector=str(candidate.get("sector", "")),
        portfolio_priority=int(order.get("_portfolio_priority", 10_000)),
        execution=_execution_or_none(candidate.get("execution")),
    )
    record["_order_sequence"] = int(order.get("_order_sequence", 0))
    return record


def build_portfolio_equity_curve(
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """按现金和持仓市值重估组合净值，保留复利再投资路径。"""
    normalized = normalize_bars(bars)
    return _build_portfolio_equity_curve_from_normalized(normalized, trades, initial_equity)


def _build_portfolio_equity_curve_from_normalized(
    normalized: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """按已标准化 K 线重估组合净值；内部热路径避免重复 normalize。"""
    if normalized.empty:
        return pd.DataFrame(columns=["date", "net_value", "cash", "position_value", "gross_exposure", "open_positions"])

    timeline = pd.Series(normalized["date"].drop_duplicates().sort_values().to_list())
    if trades.empty:
        return pd.DataFrame(
            {
                "date": timeline,
                "net_value": initial_equity,
                "cash": initial_equity,
                "position_value": 0.0,
                "gross_exposure": 0.0,
                "open_positions": 0,
            }
        )

    close_matrix = (
        normalized.pivot_table(index="date", columns="stock_code", values="close", aggfunc="last")
        .sort_index()
        .ffill()
    )
    entries_by_date = _portfolio_entries_by_date(trades)
    cash = float(initial_equity)
    positions: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for current_time in timeline:
        cash, positions = _settle_exited_positions(cash, positions, pd.Timestamp(current_time))
        position_value_before_entries = _marked_position_value(positions, close_matrix, pd.Timestamp(current_time))
        equity_before_entries = cash + position_value_before_entries
        for trade in entries_by_date.get(pd.Timestamp(current_time), []):
            allocation = equity_before_entries * float(trade["capital_fraction"])
            if allocation <= 0:
                continue
            position = _new_position(trade, allocation)
            cash += _entry_cash_delta(position)
            if pd.Timestamp(position["exit_date"]) <= pd.Timestamp(current_time):
                cash += _position_exit_cash_delta(position)
            else:
                positions.append(position)
        position_value = _marked_position_value(positions, close_matrix, pd.Timestamp(current_time))
        net_value = cash + position_value
        gross_exposure = _gross_exposure(positions, close_matrix, pd.Timestamp(current_time), net_value)
        records.append(
            {
                "date": current_time,
                "net_value": float(net_value),
                "cash": float(cash),
                "position_value": float(position_value),
                "gross_exposure": float(gross_exposure),
                "open_positions": int(len(positions)),
            }
        )
    return pd.DataFrame(records)


def _portfolio_entries_by_date(trades: pd.DataFrame) -> dict[pd.Timestamp, list[dict[str, object]]]:
    """按入场时间组织净值重估所需字段，避免把整张成交表转成 records。"""
    sorted_trades = trades.sort_values(["entry_date", "portfolio_priority", "stock_code"], kind="mergesort")
    entries: dict[pd.Timestamp, list[dict[str, object]]] = {}
    for entry_date, stock_code, side, entry_price, exit_date, raw_return_pct, capital_fraction in zip(
        pd.to_datetime(sorted_trades["entry_date"], errors="coerce"),
        sorted_trades["stock_code"].astype(str),
        sorted_trades["side"].fillna("long").astype(str),
        pd.to_numeric(sorted_trades["entry_price"], errors="coerce").fillna(0.0),
        pd.to_datetime(sorted_trades["exit_date"], errors="coerce"),
        pd.to_numeric(sorted_trades["raw_return_pct"], errors="coerce").fillna(0.0),
        pd.to_numeric(sorted_trades["capital_fraction"], errors="coerce").fillna(0.0),
        strict=True,
    ):
        if pd.isna(entry_date):
            continue
        entries.setdefault(pd.Timestamp(entry_date), []).append(
            {
                "stock_code": stock_code,
                "side": side,
                "entry_price": float(entry_price),
                "exit_date": pd.Timestamp(exit_date),
                "raw_return_pct": float(raw_return_pct),
                "capital_fraction": float(capital_fraction),
            }
        )
    return entries


def _new_position(trade: Mapping[str, object], allocation: float) -> dict[str, object]:
    return {
        "stock_code": str(trade["stock_code"]),
        "side": str(trade.get("side", "long")),
        "entry_price": float(trade["entry_price"]),
        "exit_date": pd.Timestamp(trade["exit_date"]),
        "raw_return_pct": float(trade["raw_return_pct"]),
        "allocation": float(allocation),
    }


def _settle_exited_positions(
    cash: float,
    positions: list[dict[str, object]],
    current_time: pd.Timestamp,
) -> tuple[float, list[dict[str, object]]]:
    remaining: list[dict[str, object]] = []
    for position in positions:
        if pd.Timestamp(position["exit_date"]) <= current_time:
            cash += _position_exit_cash_delta(position)
        else:
            remaining.append(position)
    return cash, remaining


def _entry_cash_delta(position: dict[str, object]) -> float:
    allocation = float(position["allocation"])
    return allocation if str(position["side"]) == "short" else -allocation


def _position_exit_cash_delta(position: dict[str, object]) -> float:
    allocation = float(position["allocation"])
    raw_return = float(position["raw_return_pct"]) / 100.0
    if str(position["side"]) == "short":
        return -allocation * (1.0 - raw_return)
    return allocation * (1.0 + raw_return)


def _marked_position_value(
    positions: list[dict[str, object]],
    close_matrix: pd.DataFrame,
    current_time: pd.Timestamp,
) -> float:
    return float(sum(_marked_position_value_one(position, close_matrix, current_time) for position in positions))


def _marked_position_value_one(
    position: dict[str, object],
    close_matrix: pd.DataFrame,
    current_time: pd.Timestamp,
) -> float:
    symbol = str(position["stock_code"])
    if current_time not in close_matrix.index or symbol not in close_matrix.columns:
        return _unmarked_position_value(position)
    mark_price = close_matrix.loc[current_time, symbol]
    if pd.isna(mark_price):
        return _unmarked_position_value(position)
    mark_ratio = float(mark_price) / float(position["entry_price"])
    allocation = float(position["allocation"])
    if str(position["side"]) == "short":
        return -allocation * mark_ratio
    return allocation * mark_ratio


def _unmarked_position_value(position: dict[str, object]) -> float:
    allocation = float(position["allocation"])
    return -allocation if str(position["side"]) == "short" else allocation


def _gross_exposure(
    positions: list[dict[str, object]],
    close_matrix: pd.DataFrame,
    current_time: pd.Timestamp,
    net_value: float,
) -> float:
    if net_value <= 0:
        return 0.0
    marked = sum(abs(_marked_position_value_one(position, close_matrix, current_time)) for position in positions)
    return float(marked / net_value)


def _sort_orders_for_simulation(orders: pd.DataFrame) -> pd.DataFrame:
    validate_order_frame_columns(orders, extra_required=("strategy_name",))
    result = orders.copy()
    result["_order_sequence"] = range(len(result))
    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce")
    return result.sort_values(["signal_date", "stock_code", "order_id", "_order_sequence"], kind="mergesort").reset_index(
        drop=True
    )


def _portfolio_order_decisions_frame(decisions: list[dict[str, object]]) -> pd.DataFrame:
    """组合决策表按信号时间稳定输出，便于按回测时间线复盘。"""
    if not decisions:
        return pd.DataFrame(columns=ORDER_DECISION_COLUMNS)
    frame = pd.DataFrame(decisions)
    for column in ORDER_DECISION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "_order_sequence" not in frame.columns:
        frame["_order_sequence"] = range(len(frame))
    frame["_signal_date_sort"] = pd.to_datetime(frame["signal_date"], errors="coerce")
    sorted_frame = frame.sort_values(
        ["_signal_date_sort", "stock_code", "order_id", "_order_sequence"],
        kind="mergesort",
        na_position="last",
    )
    return sorted_frame.loc[:, ORDER_DECISION_COLUMNS].reset_index(drop=True)


def _order_with_portfolio_priority(order: object, pcfg: PortfolioConfig) -> pd.Series:
    result = order.copy() if isinstance(order, pd.Series) else pd.Series(dict(order))
    result["_portfolio_priority"] = int(pcfg.strategy_priority.get(str(result.get("strategy_name", "")), 10_000))
    return result


def _has_symbol_overlap(open_positions: list[dict[str, object]], symbol: str, pcfg: PortfolioConfig) -> bool:
    if pcfg.allow_same_symbol_overlap:
        return False
    return any(item["stock_code"] == symbol for item in open_positions)


def _next_capital_fraction(
    open_positions: list[dict[str, object]],
    order: pd.Series,
    sector: str,
    risk_fraction: float,
    pcfg: PortfolioConfig,
) -> float:
    max_capital = 1.0 - pcfg.reserve_cash
    margin_rate = _order_margin_rate(order, pcfg)
    used_margin = sum(float(item["margin_fraction"]) for item in open_positions)
    available_margin = max_capital - used_margin
    base = _base_capital_fraction(risk_fraction, max_capital, margin_rate, pcfg)
    strategy_room = _remaining_named_limit(open_positions, "strategy_name", str(order["strategy_name"]), pcfg.strategy_capital_limit)
    sector_room = _remaining_named_limit(open_positions, "sector", sector, pcfg.sector_capital_limit)
    room_by_margin = min(available_margin, strategy_room, sector_room) / margin_rate
    return float(round(max(0.0, min(base, room_by_margin)), 12))


def _base_capital_fraction(
    risk_fraction: float,
    max_capital: float,
    margin_rate: float,
    pcfg: PortfolioConfig,
) -> float:
    max_trade_notional = pcfg.max_capital_per_trade / margin_rate
    if pcfg.risk_per_trade is not None and risk_fraction > 0:
        return min(pcfg.risk_per_trade / risk_fraction, max_trade_notional)
    if pcfg.capital_per_trade is not None:
        return min(pcfg.capital_per_trade / margin_rate, max_trade_notional)
    return min(max_capital / pcfg.max_open_positions / margin_rate, max_trade_notional)


def _order_margin_fraction(order: pd.Series, capital_fraction: float, pcfg: PortfolioConfig) -> float:
    return float(capital_fraction * _order_margin_rate(order, pcfg))


def _order_margin_rate(order: pd.Series, pcfg: PortfolioConfig) -> float:
    return pcfg.short_margin_rate if str(order.get("side", "long")) == "short" else 1.0


def _execution_or_none(value: object) -> OrderExecutionResult | None:
    return value if isinstance(value, OrderExecutionResult) else None


def _remaining_named_limit(
    open_positions: list[dict[str, object]],
    field: str,
    value: str,
    limits: Mapping[str, float],
) -> float:
    if value not in limits:
        return 1.0
    used = sum(float(item["margin_fraction"]) for item in open_positions if str(item.get(field, "")) == value)
    return float(limits[value] - used)


def _trade_risk_fraction(trade: Mapping[str, object]) -> float:
    entry_price = float(trade["entry_price"])
    risk_per_share = float(trade["risk_per_share"])
    if entry_price <= 0 or risk_per_share <= 0:
        return 0.0
    return float(risk_per_share / entry_price)


def _order_sector(order: pd.Series, pcfg: PortfolioConfig) -> str:
    metadata = order.get("metadata", {})
    if isinstance(metadata, Mapping):
        sector = metadata.get(pcfg.sector_metadata_key, pcfg.default_sector)
    else:
        sector = pcfg.default_sector
    text = str(sector).strip()
    if text and text != pcfg.default_sector:
        return text
    mapped = pcfg.symbol_sector_map.get(normalize_symbol(str(order.get("stock_code", ""))), "")
    mapped_text = str(mapped).strip()
    return mapped_text or text or pcfg.default_sector


def _validate_portfolio_config(pcfg: PortfolioConfig) -> None:
    if pcfg.max_open_positions < 1:
        raise ValueError("max_open_positions 至少需要 1。")
    if pcfg.capital_per_trade is not None and not 0 < pcfg.capital_per_trade <= 1:
        raise ValueError("capital_per_trade 必须在 0 到 1 之间。")
    if pcfg.risk_per_trade is not None and not 0 < pcfg.risk_per_trade <= 1:
        raise ValueError("risk_per_trade 必须在 0 到 1 之间。")
    if not 0 < pcfg.max_capital_per_trade <= 1:
        raise ValueError("max_capital_per_trade 必须在 0 到 1 之间。")
    if pcfg.short_margin_rate <= 0:
        raise ValueError("short_margin_rate 必须大于 0。")
    if not 0 <= pcfg.reserve_cash < 1:
        raise ValueError("reserve_cash 必须在 0 到 1 之间。")
    for name, value in {**pcfg.strategy_capital_limit, **pcfg.sector_capital_limit}.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 的资金上限必须在 0 到 1 之间。")
