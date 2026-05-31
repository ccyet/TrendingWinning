from __future__ import annotations

import math

import pandas as pd

from trending_winning.backtest.returns import downside_deviation


EQUITY_RETURN_STAT_KEYS = (
    "total_return",
    "equity_return_std",
    "equity_sharpe",
    "equity_sortino",
    "annualized_return",
    "annualized_volatility",
    "annualized_sharpe",
    "annualized_sortino",
    "calmar_ratio",
)


def equity_return_statistics(
    equity_curve: pd.DataFrame,
    net_value: pd.Series,
    *,
    max_drawdown: float,
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """从已排序净值序列计算收益、波动和年化指标。"""
    clean = pd.to_numeric(net_value, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if clean.empty:
        return empty_equity_return_statistics()

    returns = clean.pct_change().dropna()
    return_std = _std_or_zero(returns)
    return_downside = downside_deviation(returns)
    total_return = _round_float(clean.iloc[-1] / clean.iloc[0] - 1.0)
    annual_periods = float(periods_per_year or infer_periods_per_year(equity_curve))
    annual_return = annualized_return(clean, annual_periods, len(returns))
    annual_volatility = _round_float(return_std * math.sqrt(annual_periods))
    return {
        "total_return": total_return,
        "equity_return_std": return_std,
        "equity_sharpe": ratio_or_zero(_round_float(returns.mean()), return_std),
        "equity_sortino": ratio_or_zero(_round_float(returns.mean()), return_downside),
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "annualized_sharpe": annualized_ratio(returns, return_std, annual_periods),
        "annualized_sortino": annualized_ratio(returns, return_downside, annual_periods),
        "calmar_ratio": ratio_or_zero(annual_return, abs(max_drawdown)),
    }


def empty_equity_return_statistics() -> dict[str, float]:
    return {key: 0.0 for key in EQUITY_RETURN_STAT_KEYS}


def infer_periods_per_year(equity_curve: pd.DataFrame) -> float:
    """按净值时间间隔估算年化周期；分钟级按 A 股 4 小时交易日换算。"""
    if "date" not in equity_curve.columns:
        return 252.0
    dates = pd.to_datetime(equity_curve["date"], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(dates) < 2:
        return 252.0
    delta_seconds = dates.diff().dropna().dt.total_seconds()
    if delta_seconds.empty:
        return 252.0
    median_seconds = float(delta_seconds.median())
    if median_seconds <= 0:
        return 252.0
    trading_day_seconds = 4.0 * 60.0 * 60.0
    if median_seconds < trading_day_seconds:
        return _round_float(252.0 * trading_day_seconds / median_seconds)
    return 252.0


def annualized_return(net_value: pd.Series, periods_per_year: float, observed_periods: int) -> float:
    """按首尾净值和观测周期计算复合年化收益。"""
    clean = pd.to_numeric(net_value, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if observed_periods <= 0 or periods_per_year <= 0 or clean.empty:
        return 0.0
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    return _round_float((end / start) ** (periods_per_year / observed_periods) - 1.0)


def annualized_ratio(returns: pd.Series, denominator: float, periods_per_year: float) -> float:
    """把单周期 Sharpe/Sortino 放大为年化比率。"""
    clean = pd.to_numeric(returns, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if clean.empty or denominator <= 0 or periods_per_year <= 0:
        return 0.0
    return _round_float(float(clean.mean()) / denominator * math.sqrt(periods_per_year))


def ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _std_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.std(ddof=0))


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
