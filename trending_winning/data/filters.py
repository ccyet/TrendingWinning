from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trending_winning.data.schema import normalize_bars, normalize_symbol


@dataclass(frozen=True)
class LimitOpenFilterConfig:
    """日 K 一字涨停过滤参数；用于剔除会污染分钟回测的交易日。"""

    tick_size: float = 0.01
    tick_tolerance: float = 0.005
    pct_tolerance: float = 0.0005


def board_limit_pct(symbol: str, board: str = "") -> float:
    normalized_symbol = normalize_symbol(symbol)
    normalized_board = str(board).strip().upper()
    code = normalized_symbol.split(".", 1)[0]
    exchange = normalized_symbol.split(".", 1)[1] if "." in normalized_symbol else ""

    if normalized_board in {"STAR", "KCB", "科创板"} or exchange == "SH" and code.startswith(("688", "689")):
        return 0.20
    if normalized_board in {"CHINEXT", "CYB", "创业板"} or exchange == "SZ" and code.startswith(("300", "301")):
        return 0.20
    if normalized_board in {"BJ", "BSE", "北交所"} or exchange == "BJ" or code.startswith(("4", "8", "920")):
        return 0.30
    return 0.10


def round_to_tick(value: float, tick_size: float = 0.01) -> float:
    if tick_size <= 0:
        raise ValueError("tick_size 必须大于 0。")
    return round(round(float(value) / tick_size) * tick_size, 10)


def limit_open_dates(
    daily_bars: pd.DataFrame,
    *,
    config: LimitOpenFilterConfig | None = None,
) -> pd.DataFrame:
    cfg = config or LimitOpenFilterConfig()
    daily = normalize_bars(daily_bars)
    if daily.empty:
        return pd.DataFrame(columns=["stock_code", "session_date", "limit_pct", "limit_up_open"])

    daily = daily.sort_values(["stock_code", "date"]).reset_index(drop=True)
    daily["session_date"] = daily["date"].dt.normalize()
    daily["_prev_close"] = daily.groupby("stock_code", sort=False)["close"].shift(1)
    daily["_limit_pct"] = daily["stock_code"].map(board_limit_pct)
    daily["_limit_up_open"] = [
        round_to_tick(prev_close * (1.0 + limit_pct), cfg.tick_size)
        if pd.notna(prev_close)
        else pd.NA
        for prev_close, limit_pct in zip(daily["_prev_close"], daily["_limit_pct"], strict=False)
    ]
    exact_pass = (daily["open"] - daily["_limit_up_open"]).abs() <= cfg.tick_tolerance
    pct_pass = daily["open"] / daily["_prev_close"] - 1.0 >= daily["_limit_pct"] - cfg.pct_tolerance
    filtered = daily.loc[(exact_pass | pct_pass).fillna(False), ["stock_code", "session_date", "_limit_pct", "_limit_up_open"]]
    return filtered.rename(columns={"_limit_pct": "limit_pct", "_limit_up_open": "limit_up_open"}).reset_index(drop=True)


def filter_limit_open_days(
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    config: LimitOpenFilterConfig | None = None,
) -> pd.DataFrame:
    intraday = normalize_bars(intraday_bars)
    if intraday.empty:
        return intraday

    blocked = limit_open_dates(daily_bars, config=config)
    if blocked.empty:
        return intraday

    result = intraday.copy()
    result["_session_date"] = result["date"].dt.normalize()
    blocked_keys = set(zip(blocked["stock_code"], blocked["session_date"], strict=False))
    keep_mask = [
        (symbol, session_date) not in blocked_keys
        for symbol, session_date in zip(result["stock_code"], result["_session_date"], strict=False)
    ]
    return result.loc[keep_mask].drop(columns=["_session_date"]).reset_index(drop=True)
