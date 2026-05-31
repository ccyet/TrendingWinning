from __future__ import annotations

import pandas as pd


EQUITY_EXPOSURE_STAT_KEYS = (
    "avg_gross_exposure",
    "max_gross_exposure",
    "avg_margin_exposure",
    "max_margin_exposure",
    "exposure_bar_ratio",
    "avg_open_positions",
    "max_open_positions",
    "avg_cash_ratio",
    "min_cash_ratio",
    "max_cash_ratio",
    "avg_net_exposure",
    "min_net_exposure",
    "max_net_exposure",
)


def equity_exposure_statistics(equity_curve: pd.DataFrame, net_value: pd.Series) -> dict[str, float]:
    """统计净值曲线上的持仓、保证金、现金和净暴露。"""
    if equity_curve.empty:
        return empty_equity_exposure_statistics()

    gross_exposure = _numeric_column(equity_curve, "gross_exposure")
    margin_exposure = _numeric_column(equity_curve, "margin_exposure")
    open_positions = _numeric_column(equity_curve, "open_positions")
    cash_ratio = ratio_column_to_net_value(equity_curve, "cash", net_value)
    net_exposure = ratio_column_to_net_value(equity_curve, "position_value", net_value)

    return {
        "avg_gross_exposure": _mean_or_zero(gross_exposure),
        "max_gross_exposure": _max_or_zero(gross_exposure),
        "avg_margin_exposure": _mean_or_zero(margin_exposure),
        "max_margin_exposure": _max_or_zero(margin_exposure),
        "exposure_bar_ratio": _round_float((gross_exposure > 0).mean()) if not gross_exposure.empty else 0.0,
        "avg_open_positions": _mean_or_zero(open_positions),
        "max_open_positions": _max_or_zero(open_positions),
        "avg_cash_ratio": _mean_or_zero(cash_ratio),
        "min_cash_ratio": _min_or_zero(cash_ratio),
        "max_cash_ratio": _max_or_zero(cash_ratio),
        "avg_net_exposure": _mean_or_zero(net_exposure),
        "min_net_exposure": _min_or_zero(net_exposure),
        "max_net_exposure": _max_or_zero(net_exposure),
    }


def empty_equity_exposure_statistics() -> dict[str, float]:
    return {key: 0.0 for key in EQUITY_EXPOSURE_STAT_KEYS}


def ratio_column_to_net_value(frame: pd.DataFrame, column: str, net_value: pd.Series) -> pd.Series:
    """把现金或持仓市值转成净值占比；净值无效时显式归零。"""
    if column not in frame.columns or net_value.empty:
        return pd.Series(dtype=float)
    length = min(len(frame), len(net_value))
    numerator = pd.to_numeric(frame[column].iloc[:length], errors="coerce").fillna(0.0).astype(float).reset_index(
        drop=True
    )
    denominator = pd.to_numeric(net_value.iloc[:length], errors="coerce").astype(float).reset_index(drop=True)
    ratio = numerator.div(denominator.where(denominator.gt(0))).replace([float("inf"), float("-inf")], pd.NA)
    return ratio.fillna(0.0).astype(float)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.max())


def _min_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.min())


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
