from __future__ import annotations

import math

import pandas as pd


RETURN_SERIES_STAT_KEYS = (
    "avg_return",
    "return_std",
    "downside_deviation",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "best_trade",
    "worst_trade",
    "return_p05",
    "return_p25",
    "return_p50",
    "return_p75",
    "return_p95",
    "cvar_95",
)


def return_series_statistics(returns: pd.Series) -> dict[str, float]:
    """统计逐笔收益分布和路径连续性，输入收益应为小数比例。"""
    clean = _clean_returns(returns)
    if clean.empty:
        return {key: 0.0 for key in RETURN_SERIES_STAT_KEYS}

    return_p05 = _quantile(clean, 0.05)
    return {
        "avg_return": _round_float(clean.mean()),
        "return_std": _round_float(clean.std(ddof=0)),
        "downside_deviation": downside_deviation(clean),
        "max_consecutive_wins": float(max_return_streak(clean, positive=True)),
        "max_consecutive_losses": float(max_return_streak(clean, positive=False)),
        "best_trade": _round_float(clean.max()),
        "worst_trade": _round_float(clean.min()),
        "return_p05": return_p05,
        "return_p25": _quantile(clean, 0.25),
        "return_p50": _quantile(clean, 0.50),
        "return_p75": _quantile(clean, 0.75),
        "return_p95": _quantile(clean, 0.95),
        "cvar_95": _round_float(clean.loc[clean <= return_p05].mean()),
    }


def downside_deviation(returns: pd.Series) -> float:
    """按全样本下行收益计算 Sortino 分母，正收益也计入样本长度。"""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0.0
    downside = clean.clip(upper=0.0)
    return _round_float(math.sqrt(float(downside.pow(2).mean())))


def max_return_streak(returns: pd.Series, *, positive: bool) -> int:
    """计算最长连续盈利或亏损笔数，0 收益会打断连续区间。"""
    clean = _clean_returns(returns)
    if clean.empty:
        return 0
    mask = clean.gt(0.0) if positive else clean.lt(0.0)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    streaks = mask.astype(int).groupby(groups).sum()
    return int(streaks.max()) if not streaks.empty else 0


def _clean_returns(returns: pd.Series) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").dropna().astype(float).reset_index(drop=True)


def _quantile(values: pd.Series, quantile: float) -> float:
    return _round_float(values.quantile(quantile))


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
