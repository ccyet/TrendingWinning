from __future__ import annotations

import math

import pandas as pd


TRADE_RISK_QUALITY_STAT_KEYS = (
    "avg_r_multiple",
    "median_r_multiple",
    "best_r_multiple",
    "worst_r_multiple",
    "r_profit_factor",
    "system_quality_number",
    "avg_mae_pct",
    "avg_mfe_pct",
    "avg_mae_r",
    "avg_mfe_r",
)


def trade_risk_quality_statistics(trades: pd.DataFrame) -> dict[str, float]:
    """统计 R 倍数和 MAE/MFE 路径质量，衡量信号执行后的风险效率。"""
    if trades.empty:
        return empty_trade_risk_quality_statistics()

    r_multiple = _optional_numeric_column(trades, "r_multiple")
    positive_r = r_multiple.loc[r_multiple > 0]
    negative_r = r_multiple.loc[r_multiple < 0]
    r_std = _std_or_zero(r_multiple)
    mae_pct = _optional_numeric_column(trades, "mae_pct")
    mfe_pct = _optional_numeric_column(trades, "mfe_pct")
    mae_r = _optional_numeric_column(trades, "mae_r")
    mfe_r = _optional_numeric_column(trades, "mfe_r")

    return {
        "avg_r_multiple": _mean_or_zero(r_multiple),
        "median_r_multiple": _median_or_zero(r_multiple),
        "best_r_multiple": _max_or_zero(r_multiple),
        "worst_r_multiple": _min_or_zero(r_multiple),
        "r_profit_factor": _ratio_or_inf(_round_float(positive_r.sum()), _round_float(abs(negative_r.sum()))),
        "system_quality_number": system_quality_number(r_multiple, r_std),
        "avg_mae_pct": _mean_or_zero(mae_pct),
        "avg_mfe_pct": _mean_or_zero(mfe_pct),
        "avg_mae_r": _mean_or_zero(mae_r),
        "avg_mfe_r": _mean_or_zero(mfe_r),
    }


def empty_trade_risk_quality_statistics() -> dict[str, float]:
    return {key: 0.0 for key in TRADE_RISK_QUALITY_STAT_KEYS}


def system_quality_number(r_multiple: pd.Series, r_std: float | None = None) -> float:
    """按 Van Tharp SQN 口径衡量 R 倍数序列质量。"""
    clean = pd.to_numeric(r_multiple, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if clean.empty:
        return 0.0
    denominator = _std_or_zero(clean) if r_std is None else float(r_std)
    if denominator <= 0:
        return 0.0
    return _round_float(math.sqrt(len(clean)) * float(clean.mean()) / denominator)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float).reset_index(drop=True)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _median_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.median())


def _std_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.std(ddof=0))


def _max_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.max())


def _min_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.min())


def _ratio_or_inf(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return float("inf") if numerator > 0 else 0.0


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
