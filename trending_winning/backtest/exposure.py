from __future__ import annotations

import pandas as pd


TRADE_EXPOSURE_STAT_KEYS = (
    "exposure_bars",
    "return_contribution",
    "return_per_exposure_bar",
    "capital_turnover",
    "avg_capital_fraction",
    "max_capital_fraction",
    "margin_turnover",
    "avg_margin_fraction",
    "max_margin_fraction",
    "capital_exposure_bars",
    "margin_exposure_bars",
    "avg_capital_exposure_per_trade",
    "avg_margin_exposure_per_trade",
    "return_per_capital_exposure_bar",
    "return_per_margin_exposure_bar",
    "capital_weighted_raw_return",
)


def trade_exposure_statistics(trades: pd.DataFrame, returns: pd.Series) -> dict[str, float]:
    """统计持仓 K 数、资金占用和单位暴露收益，供单策略和组合拆分复用。"""
    clean_returns = pd.to_numeric(returns, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if clean_returns.empty:
        return empty_trade_exposure_statistics()

    holding_bars = _numeric_column(trades, "holding_bars").iloc[: len(clean_returns)].reset_index(drop=True)
    capital_fraction = _optional_numeric_column(trades, "capital_fraction").iloc[: len(clean_returns)].reset_index(
        drop=True
    )
    margin_fraction = _optional_numeric_column(trades, "margin_fraction").iloc[: len(clean_returns)].reset_index(
        drop=True
    )
    raw_returns = (
        _optional_numeric_column(trades, "raw_return_pct").iloc[: len(clean_returns)].reset_index(drop=True) / 100.0
        if "raw_return_pct" in trades.columns
        else clean_returns
    )
    capital_exposure = weighted_exposure_bars(holding_bars, capital_fraction)
    margin_exposure = weighted_exposure_bars(holding_bars, margin_fraction)
    exposure_bar_count = _round_float(holding_bars.sum())
    return_contribution = _round_float(clean_returns.sum())
    capital_exposure_bars = _round_float(capital_exposure.sum()) if not capital_exposure.empty else 0.0
    margin_exposure_bars = _round_float(margin_exposure.sum()) if not margin_exposure.empty else 0.0

    return {
        "exposure_bars": exposure_bar_count,
        "return_contribution": return_contribution,
        "return_per_exposure_bar": _ratio_or_zero(return_contribution, exposure_bar_count),
        "capital_turnover": _round_float(capital_fraction.sum()) if not capital_fraction.empty else 0.0,
        "avg_capital_fraction": _mean_or_zero(capital_fraction),
        "max_capital_fraction": _max_or_zero(capital_fraction),
        "margin_turnover": _round_float(margin_fraction.sum()) if not margin_fraction.empty else 0.0,
        "avg_margin_fraction": _mean_or_zero(margin_fraction),
        "max_margin_fraction": _max_or_zero(margin_fraction),
        "capital_exposure_bars": capital_exposure_bars,
        "margin_exposure_bars": margin_exposure_bars,
        "avg_capital_exposure_per_trade": _mean_or_zero(capital_exposure),
        "avg_margin_exposure_per_trade": _mean_or_zero(margin_exposure),
        "return_per_capital_exposure_bar": _ratio_or_zero(return_contribution, capital_exposure_bars),
        "return_per_margin_exposure_bar": _ratio_or_zero(return_contribution, margin_exposure_bars),
        "capital_weighted_raw_return": weighted_mean_or_zero(raw_returns, capital_fraction),
    }


def empty_trade_exposure_statistics() -> dict[str, float]:
    return {key: 0.0 for key in TRADE_EXPOSURE_STAT_KEYS}


def weighted_exposure_bars(holding_bars: pd.Series, fractions: pd.Series) -> pd.Series:
    """资金或保证金占用乘以持仓 K 数，用于衡量长期占资压力。"""
    if holding_bars.empty or fractions.empty:
        return pd.Series(dtype=float)
    length = min(len(holding_bars), len(fractions))
    return holding_bars.iloc[:length].reset_index(drop=True) * fractions.iloc[:length].reset_index(drop=True)


def weighted_mean_or_zero(values: pd.Series, weights: pd.Series) -> float:
    if values.empty or weights.empty:
        return 0.0
    length = min(len(values), len(weights))
    value_slice = values.iloc[:length].reset_index(drop=True)
    weight_slice = weights.iloc[:length].reset_index(drop=True)
    denominator = float(weight_slice.sum())
    if denominator <= 0:
        return 0.0
    return _round_float(float((value_slice * weight_slice).sum()) / denominator)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.max())


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
