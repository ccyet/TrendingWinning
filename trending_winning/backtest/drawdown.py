from __future__ import annotations

import math

import pandas as pd


DRAWDOWN_STAT_KEYS = (
    "max_drawdown",
    "max_drawdown_duration",
    "max_drawdown_start_at",
    "max_drawdown_trough_at",
    "max_drawdown_recovery_at",
    "current_drawdown",
    "current_underwater_bars",
    "avg_drawdown",
    "ulcer_index",
    "time_under_water_ratio",
)


def equity_drawdown_statistics(data: pd.DataFrame, net_value: pd.Series) -> dict[str, object]:
    """从已排序净值序列计算完整回撤统计，供单策略和组合回测复用。"""
    numeric = pd.to_numeric(net_value, errors="coerce")
    valid = numeric.notna()
    clean = numeric.loc[valid].reset_index(drop=True)
    if clean.empty:
        return empty_drawdown_statistics()

    if not data.empty and len(data) == len(numeric):
        aligned_data = data.loc[valid.to_numpy()].reset_index(drop=True)
    else:
        aligned_data = data.iloc[: len(clean)].reset_index(drop=True) if not data.empty else pd.DataFrame(index=clean.index)
    drawdown = clean / clean.cummax() - 1.0
    result = empty_drawdown_statistics()
    result.update(
        {
            "max_drawdown": _round_float(float(drawdown.min())),
            "max_drawdown_duration": float(max_drawdown_duration(clean)),
            "current_drawdown": _round_float(float(drawdown.iloc[-1])),
            "current_underwater_bars": float(trailing_underwater_length(drawdown)),
            "avg_drawdown": _mean_or_zero(drawdown),
            "ulcer_index": _round_float(math.sqrt(float(drawdown.pow(2).mean()))) if not drawdown.empty else 0.0,
            "time_under_water_ratio": _round_float(float(drawdown.lt(0).mean())) if not drawdown.empty else 0.0,
        }
    )
    result.update(drawdown_episode_labels(aligned_data, clean, drawdown))
    return result


def drawdown_episode_labels(data: pd.DataFrame, net_value: pd.Series, drawdown: pd.Series) -> dict[str, object]:
    """定位最大回撤的起点、触底点和首次修复点。"""
    result = {
        "max_drawdown_start_at": "",
        "max_drawdown_trough_at": "",
        "max_drawdown_recovery_at": "",
    }
    if net_value.empty or drawdown.empty or float(drawdown.min()) >= 0:
        return result

    trough_pos = int(drawdown.idxmin())
    peak_value = float(net_value.cummax().iloc[trough_pos])
    prior_values = net_value.iloc[: trough_pos + 1]
    peak_positions = prior_values.index[prior_values.eq(peak_value)]
    peak_pos = int(peak_positions[-1]) if len(peak_positions) else trough_pos
    after_trough = net_value.iloc[trough_pos + 1 :]
    recovered_positions = after_trough.index[after_trough.ge(peak_value)]

    result["max_drawdown_start_at"] = equity_point_label(data, peak_pos)
    result["max_drawdown_trough_at"] = equity_point_label(data, trough_pos)
    if len(recovered_positions):
        result["max_drawdown_recovery_at"] = equity_point_label(data, int(recovered_positions[0]))
    return result


def empty_drawdown_statistics() -> dict[str, object]:
    """返回固定字段，避免无成交或空净值时输出结构漂移。"""
    return {
        "max_drawdown": 0.0,
        "max_drawdown_duration": 0.0,
        "max_drawdown_start_at": "",
        "max_drawdown_trough_at": "",
        "max_drawdown_recovery_at": "",
        "current_drawdown": 0.0,
        "current_underwater_bars": 0.0,
        "avg_drawdown": 0.0,
        "ulcer_index": 0.0,
        "time_under_water_ratio": 0.0,
    }


def trailing_underwater_length(drawdown: pd.Series) -> int:
    """统计当前连续处于水下的净值点数量。"""
    count = 0
    for value in reversed(drawdown.tolist()):
        if pd.isna(value) or float(value) >= 0:
            break
        count += 1
    return count


def max_drawdown_duration(equity: pd.Series) -> int:
    """统计最长连续水下净值点数量。"""
    peak = -math.inf
    current = 0
    best = 0
    for value in pd.to_numeric(equity, errors="coerce").dropna():
        if value >= peak:
            peak = float(value)
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def equity_point_label(data: pd.DataFrame, position: int) -> str:
    """把净值点位置转成人能读的日期或交易编号。"""
    if position < 0 or position >= len(data):
        return ""
    row = data.iloc[position]
    if "date" in data.columns:
        timestamp = pd.to_datetime(row["date"], errors="coerce")
        if pd.notna(timestamp):
            return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    if "trade_no" in data.columns and pd.notna(row["trade_no"]):
        return _compact_numeric_label(row["trade_no"])
    return str(position)


def _compact_numeric_label(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return str(value)


def _mean_or_zero(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return _round_float(values.mean())


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
