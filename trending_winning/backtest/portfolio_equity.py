from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_portfolio_equity_curve_from_normalized(
    normalized: pd.DataFrame,
    trades: pd.DataFrame,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    """按已标准化 K 线逐 K 重估组合现金、持仓市值和暴露。"""
    equity_columns = ["date", "net_value", "cash", "position_value", "gross_exposure", "margin_exposure", "open_positions"]
    if normalized.empty:
        return pd.DataFrame(columns=equity_columns)

    timeline = pd.Series(normalized["date"].drop_duplicates().sort_values().to_list())
    if trades.empty:
        return pd.DataFrame(
            {
                "date": timeline,
                "net_value": initial_equity,
                "cash": initial_equity,
                "position_value": 0.0,
                "gross_exposure": 0.0,
                "margin_exposure": 0.0,
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
        timestamp = pd.Timestamp(current_time)
        cash, positions = _settle_exited_positions(cash, positions, timestamp)
        position_value_before_entries = _marked_position_value(positions, close_matrix, timestamp)
        equity_before_entries = cash + position_value_before_entries
        for trade in entries_by_date.get(timestamp, []):
            allocation = equity_before_entries * float(trade["capital_fraction"])
            if allocation <= 0:
                continue
            position = _new_position(trade, allocation)
            cash += _entry_cash_delta(position)
            if pd.Timestamp(position["exit_date"]) <= timestamp:
                cash += _position_exit_cash_delta(position)
            else:
                positions.append(position)
        position_value = _marked_position_value(positions, close_matrix, timestamp)
        net_value = cash + position_value
        records.append(
            {
                "date": current_time,
                "net_value": float(net_value),
                "cash": float(cash),
                "position_value": float(position_value),
                "gross_exposure": _gross_exposure(positions, close_matrix, timestamp, net_value),
                "margin_exposure": _margin_exposure(positions, close_matrix, timestamp, net_value),
                "open_positions": int(len(positions)),
            }
        )
    return pd.DataFrame(records)


def _portfolio_entries_by_date(trades: pd.DataFrame) -> dict[pd.Timestamp, list[dict[str, object]]]:
    """按入场时间组织净值重估所需字段，避免把整张成交表转成 records。"""
    sorted_trades = trades.sort_values(["entry_date", "portfolio_priority", "stock_code"], kind="mergesort")
    capital_fraction_values = pd.to_numeric(sorted_trades["capital_fraction"], errors="coerce").fillna(0.0)
    margin_fraction_values = _portfolio_margin_fraction_values(sorted_trades, capital_fraction_values)
    entries: dict[pd.Timestamp, list[dict[str, object]]] = {}
    for entry_date, stock_code, side, entry_price, exit_date, raw_return_pct, capital_fraction, margin_fraction in zip(
        pd.to_datetime(sorted_trades["entry_date"], errors="coerce"),
        sorted_trades["stock_code"].astype(str),
        sorted_trades["side"].fillna("long").astype(str),
        pd.to_numeric(sorted_trades["entry_price"], errors="coerce").fillna(0.0),
        pd.to_datetime(sorted_trades["exit_date"], errors="coerce"),
        pd.to_numeric(sorted_trades["raw_return_pct"], errors="coerce").fillna(0.0),
        capital_fraction_values,
        margin_fraction_values,
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
                "margin_rate": _entry_margin_rate(float(capital_fraction), float(margin_fraction)),
            }
        )
    return entries


def _portfolio_margin_fraction_values(sorted_trades: pd.DataFrame, capital_fraction_values: pd.Series) -> pd.Series:
    if "margin_fraction" not in sorted_trades.columns:
        return capital_fraction_values
    return pd.to_numeric(sorted_trades["margin_fraction"], errors="coerce").fillna(capital_fraction_values)


def _new_position(trade: Mapping[str, object], allocation: float) -> dict[str, object]:
    return {
        "stock_code": str(trade["stock_code"]),
        "side": str(trade.get("side", "long")),
        "entry_price": float(trade["entry_price"]),
        "exit_date": pd.Timestamp(trade["exit_date"]),
        "raw_return_pct": float(trade["raw_return_pct"]),
        "allocation": float(allocation),
        "margin_rate": float(trade.get("margin_rate", 1.0)),
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


def _margin_exposure(
    positions: list[dict[str, object]],
    close_matrix: pd.DataFrame,
    current_time: pd.Timestamp,
    net_value: float,
) -> float:
    if net_value <= 0:
        return 0.0
    marked_margin = sum(
        abs(_marked_position_value_one(position, close_matrix, current_time)) * float(position.get("margin_rate", 1.0))
        for position in positions
    )
    return float(marked_margin / net_value)


def _entry_margin_rate(capital_fraction: float, margin_fraction: float) -> float:
    if capital_fraction <= 0 or margin_fraction <= 0:
        return 1.0
    return float(margin_fraction / capital_fraction)
