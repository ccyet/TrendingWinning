from __future__ import annotations

import math

import pandas as pd


PERIOD_STAT_KEYS = [
    "count",
    "win_rate",
    "positive_count",
    "negative_count",
    "avg_return",
    "return_std",
    "best_return",
    "best_return_period",
    "worst_return",
    "worst_return_period",
    "avg_drawdown",
    "worst_drawdown",
    "worst_drawdown_period",
    "avg_observation_count",
    "max_consecutive_gains",
    "max_consecutive_losses",
    "max_recovery_periods",
    "underwater_ratio",
    "current_underwater_periods",
]

PERIOD_RETURN_COLUMNS = pd.Index(
    ["period", "start", "end", "start_net_value", "end_net_value", "return", "max_drawdown", "observation_count"]
)


def compute_period_returns(equity_curve: pd.DataFrame, *, freq: str = "M") -> pd.DataFrame:
    """按自然周期拆分净值收益；用于月度/年度复盘和策略稳定性检查。"""
    if equity_curve.empty or not {"date", "net_value"}.issubset(equity_curve.columns):
        return pd.DataFrame(columns=PERIOD_RETURN_COLUMNS)

    columns = ["date", "net_value", *(["trade_no"] if "trade_no" in equity_curve.columns else [])]
    data = equity_curve[columns].copy()
    data["_row_order"] = range(len(data))
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["net_value"] = pd.to_numeric(data["net_value"], errors="coerce")
    data = data.dropna(subset=["date", "net_value"])
    sort_columns = ["date", "_row_order"]
    if "trade_no" in data.columns:
        data["trade_no"] = pd.to_numeric(data["trade_no"], errors="coerce")
        sort_columns = ["date", "trade_no", "_row_order"]
    data = data.sort_values(sort_columns, kind="mergesort")
    if data.empty:
        return pd.DataFrame(columns=PERIOD_RETURN_COLUMNS)

    data["_period"] = data["date"].dt.to_period(freq)
    rows: list[dict[str, object]] = []
    previous_net_value: float | None = None
    for period, group in data.groupby("_period", sort=True):
        first_net_value = float(group.iloc[0]["net_value"])
        start_net_value = first_net_value if previous_net_value is None else float(previous_net_value)
        end_net_value = float(group.iloc[-1]["net_value"])
        period_equity = pd.concat(
            [pd.Series([start_net_value]), group["net_value"].reset_index(drop=True)],
            ignore_index=True,
        )
        period_drawdown = period_equity / period_equity.cummax() - 1.0
        rows.append(
            {
                "period": str(period),
                "start": group.iloc[0]["date"],
                "end": group.iloc[-1]["date"],
                "start_net_value": start_net_value,
                "end_net_value": end_net_value,
                "return": _ratio_or_zero(end_net_value - start_net_value, start_net_value),
                "max_drawdown": _round_float(period_drawdown.min()),
                "observation_count": int(len(group)),
            }
        )
        previous_net_value = end_net_value
    return pd.DataFrame(rows, columns=PERIOD_RETURN_COLUMNS)


def compute_period_return_statistics(period_returns: pd.DataFrame, *, prefix: str = "period") -> dict[str, object]:
    """把月度/年度收益表压成稳定性摘要；只依赖周期收益表，和策略模块完全解耦。"""
    if period_returns.empty:
        return _empty_period_return_statistics(prefix)

    returns = _numeric_column(period_returns, "return")
    drawdown = _numeric_column(period_returns, "max_drawdown")
    observation_count = _numeric_column(period_returns, "observation_count")
    period_net_value = _period_net_value_series(period_returns, returns)
    period_underwater = period_net_value < period_net_value.cummax()
    period_count = float(len(returns))
    positive_count = float((returns > 0).sum())
    negative_count = float((returns < 0).sum())
    return {
        f"{prefix}_count": period_count,
        f"{prefix}_win_rate": _ratio_or_zero(positive_count, period_count),
        f"{prefix}_positive_count": positive_count,
        f"{prefix}_negative_count": negative_count,
        f"{prefix}_avg_return": _mean_or_zero(returns),
        f"{prefix}_return_std": _std_or_zero(returns),
        f"{prefix}_best_return": _max_or_zero(returns),
        f"{prefix}_best_return_period": _period_label_at(period_returns, _series_max_position(returns)),
        f"{prefix}_worst_return": _min_or_zero(returns),
        f"{prefix}_worst_return_period": _period_label_at(period_returns, _series_min_position(returns)),
        f"{prefix}_avg_drawdown": _mean_or_zero(drawdown),
        f"{prefix}_worst_drawdown": _min_or_zero(drawdown),
        f"{prefix}_worst_drawdown_period": _period_label_at(period_returns, _series_min_position(drawdown)),
        f"{prefix}_avg_observation_count": _mean_or_zero(observation_count),
        f"{prefix}_max_consecutive_gains": float(_max_streak(returns, positive=True)),
        f"{prefix}_max_consecutive_losses": float(_max_streak(returns, positive=False)),
        f"{prefix}_max_recovery_periods": float(_max_drawdown_duration(period_net_value)),
        f"{prefix}_underwater_ratio": _round_float(float(period_underwater.mean())) if not period_underwater.empty else 0.0,
        f"{prefix}_current_underwater_periods": float(_trailing_true_length(period_underwater)),
    }


def _empty_period_return_statistics(prefix: str) -> dict[str, object]:
    result: dict[str, object] = {f"{prefix}_{key}": 0.0 for key in PERIOD_STAT_KEYS}
    for key in ("best_return_period", "worst_return_period", "worst_drawdown_period"):
        result[f"{prefix}_{key}"] = ""
    return result


def _period_net_value_series(period_returns: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """优先使用真实期末净值；没有期末净值时用周期收益复原一条相对净值。"""
    if "end_net_value" in period_returns.columns:
        return _numeric_column(period_returns, "end_net_value")
    if returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns).cumprod().reset_index(drop=True)


def _period_label_at(period_returns: pd.DataFrame, position: int | None) -> str:
    if position is None or position < 0 or position >= len(period_returns):
        return ""
    row = period_returns.iloc[position]
    if "period" in period_returns.columns and pd.notna(row["period"]):
        return str(row["period"])
    if "start" in period_returns.columns:
        timestamp = pd.to_datetime(row["start"], errors="coerce")
        if pd.notna(timestamp):
            return timestamp.strftime("%Y-%m-%d")
    return str(position)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0.0] * len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float).reset_index(drop=True)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


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


def _series_max_position(values: pd.Series) -> int | None:
    if values.empty:
        return None
    return int(values.idxmax())


def _series_min_position(values: pd.Series) -> int | None:
    if values.empty:
        return None
    return int(values.idxmin())


def _ratio_or_zero(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return _round_float(numerator / denominator)
    return 0.0


def _max_streak(returns: pd.Series, *, positive: bool) -> int:
    best = 0
    current = 0
    for value in returns:
        is_match = value > 0 if positive else value < 0
        if is_match:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_drawdown_duration(equity: pd.Series) -> int:
    peak = -math.inf
    current = 0
    best = 0
    for value in equity:
        if value >= peak:
            peak = float(value)
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def _trailing_true_length(mask: pd.Series) -> int:
    count = 0
    for value in reversed(mask.tolist()):
        if not bool(value):
            break
        count += 1
    return count


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
