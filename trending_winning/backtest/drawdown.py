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

DRAWDOWN_EPISODE_COLUMNS = pd.Index(
    [
        "episode_rank",
        "episode_no",
        "start_at",
        "trough_at",
        "recovery_at",
        "peak_net_value",
        "trough_net_value",
        "depth",
        "underwater_bars",
        "recovery_bars",
        "recovered",
    ]
)

DRAWDOWN_CURVE_COLUMNS = pd.Index(
    [
        "date",
        "trade_no",
        "net_value",
        "drawdown_net_value",
        "path_net_value",
        "drawdown",
        "point_type",
    ]
)


def price_path_drawdown_inputs(data: pd.DataFrame, net_value: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """生成回撤专用路径：每根 K 先放不利估值，再放结算净值。"""
    path = price_path_drawdown_frame(data, net_value)
    if path.empty:
        numeric_net = pd.to_numeric(net_value, errors="coerce").reset_index(drop=True)
        return data, numeric_net
    path_value = pd.to_numeric(path["_drawdown_path_value"], errors="coerce").reset_index(drop=True)
    aligned = path.drop(columns=["_drawdown_path_value", "_drawdown_point_type"], errors="ignore").reset_index(drop=True)
    return aligned, path_value


def price_path_drawdown_frame(data: pd.DataFrame, net_value: pd.Series) -> pd.DataFrame:
    """展开回撤计算路径，并标记不利价格点和结算点。"""
    numeric_net = pd.to_numeric(net_value, errors="coerce").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=[*data.columns, "_drawdown_path_value", "_drawdown_point_type"])

    aligned_data = data.reset_index(drop=True)
    drawdown_value = _drawdown_value_series(aligned_data, numeric_net)
    rows: list[dict[str, object]] = []
    for pos in range(min(len(aligned_data), len(numeric_net))):
        row = aligned_data.iloc[pos].to_dict()
        net = numeric_net.iloc[pos]
        adverse = drawdown_value.iloc[pos] if pos < len(drawdown_value) else net
        if pd.notna(net) and not _drawdown_path_needs_settlement_point(net, adverse):
            rows.append(_drawdown_path_row(row, float(net), "settlement"))
            continue
        if pd.notna(adverse):
            rows.append(_drawdown_path_row(row, float(adverse), "adverse_price"))
        if pd.notna(net):
            rows.append(_drawdown_path_row(row, float(net), "settlement"))

    if not rows:
        return pd.DataFrame(columns=[*aligned_data.columns, "_drawdown_path_value", "_drawdown_point_type"])
    return pd.DataFrame(rows)


def drawdown_curve(data: pd.DataFrame) -> pd.DataFrame:
    """输出可复核的逐点回撤曲线，区分盘中不利估值和结算净值。"""
    if data.empty or "net_value" not in data.columns:
        return pd.DataFrame(columns=DRAWDOWN_CURVE_COLUMNS)

    path = price_path_drawdown_frame(data, data["net_value"])
    if path.empty:
        return pd.DataFrame(columns=DRAWDOWN_CURVE_COLUMNS)

    path_value = pd.to_numeric(path["_drawdown_path_value"], errors="coerce")
    valid = path_value.notna()
    path = path.loc[valid].reset_index(drop=True)
    path_value = path_value.loc[valid].reset_index(drop=True)
    if path.empty:
        return pd.DataFrame(columns=DRAWDOWN_CURVE_COLUMNS)

    running_peak = path_value.cummax()
    result = pd.DataFrame(
        {
            "date": _optional_datetime_column(path, "date"),
            "trade_no": _optional_numeric_column(path, "trade_no"),
            "net_value": _optional_numeric_column(path, "net_value"),
            "drawdown_net_value": _optional_numeric_column(path, "drawdown_net_value", fallback=path["net_value"]),
            "path_net_value": path_value.astype(float),
            "drawdown": path_value / running_peak - 1.0,
            "point_type": path["_drawdown_point_type"].astype(str),
        }
    )
    return result.reindex(columns=DRAWDOWN_CURVE_COLUMNS)


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
    bar_drawdown = _bar_worst_drawdown_series(aligned_data, drawdown)
    result = empty_drawdown_statistics()
    result.update(
        {
            "max_drawdown": _round_float(float(drawdown.min())),
            "max_drawdown_duration": float(_max_drawdown_bar_duration(aligned_data, clean)),
            "current_drawdown": _round_float(float(drawdown.iloc[-1])),
            "current_underwater_bars": float(_trailing_underwater_bar_length(aligned_data, drawdown)),
            "avg_drawdown": _mean_or_zero(bar_drawdown),
            "ulcer_index": _round_float(math.sqrt(float(bar_drawdown.pow(2).mean()))) if not bar_drawdown.empty else 0.0,
            "time_under_water_ratio": _bar_underwater_ratio(aligned_data, drawdown),
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


def drawdown_episodes(data: pd.DataFrame, net_value: pd.Series, *, limit: int | None = None) -> pd.DataFrame:
    """拆分每段水下区间，按回撤深度输出可复盘的起点、触底和修复点。"""
    aligned_data, clean = _aligned_drawdown_inputs(data, net_value)
    if clean.empty:
        return pd.DataFrame(columns=DRAWDOWN_EPISODE_COLUMNS)

    episodes: list[dict[str, object]] = []
    peak_pos = 0
    peak_value = float(clean.iloc[0])
    current: dict[str, object] | None = None
    episode_no = 0

    for pos, raw_value in enumerate(clean.iloc[1:].tolist(), start=1):
        value = float(raw_value)
        if value >= peak_value:
            if current is not None:
                _finalize_drawdown_episode(current, aligned_data, clean, recovery_pos=pos)
                episodes.append(current)
                current = None
            peak_pos = pos
            peak_value = value
            continue

        if current is None:
            episode_no += 1
            current = _new_drawdown_episode(episode_no, peak_pos, peak_value, pos, value)
        else:
            if value < float(current["trough_net_value"]):
                current["trough_pos"] = pos
                current["trough_net_value"] = value

    if current is not None:
        _finalize_drawdown_episode(current, aligned_data, clean, recovery_pos=None)
        episodes.append(current)

    if not episodes:
        return pd.DataFrame(columns=DRAWDOWN_EPISODE_COLUMNS)

    rows = [_public_drawdown_episode_row(episode, aligned_data) for episode in episodes]
    result = pd.DataFrame(rows)
    result = result.sort_values(["depth", "start_at", "episode_no"], kind="mergesort").reset_index(drop=True)
    if limit is not None:
        result = result.head(max(int(limit), 0)).reset_index(drop=True)
    result.insert(0, "episode_rank", range(1, len(result) + 1))
    result["recovered"] = result["recovered"].astype(object)
    return result.reindex(columns=DRAWDOWN_EPISODE_COLUMNS)


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


def _aligned_drawdown_inputs(data: pd.DataFrame, net_value: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    numeric = pd.to_numeric(net_value, errors="coerce")
    valid = numeric.notna()
    clean = numeric.loc[valid].reset_index(drop=True).astype(float)
    if clean.empty:
        return pd.DataFrame(index=clean.index), clean
    if not data.empty and len(data) == len(numeric):
        aligned_data = data.loc[valid.to_numpy()].reset_index(drop=True)
    else:
        aligned_data = data.iloc[: len(clean)].reset_index(drop=True) if not data.empty else pd.DataFrame(index=clean.index)
    return aligned_data, clean


def _new_drawdown_episode(
    episode_no: int,
    peak_pos: int,
    peak_value: float,
    trough_pos: int,
    trough_value: float,
) -> dict[str, object]:
    return {
        "episode_no": int(episode_no),
        "peak_pos": int(peak_pos),
        "peak_net_value": float(peak_value),
        "trough_pos": int(trough_pos),
        "trough_net_value": float(trough_value),
        "underwater_bars": 1,
        "recovery_pos": None,
    }


def _finalize_drawdown_episode(
    episode: dict[str, object],
    data: pd.DataFrame,
    net_value: pd.Series,
    *,
    recovery_pos: int | None,
) -> None:
    peak_value = float(episode["peak_net_value"])
    trough_value = float(episode["trough_net_value"])
    peak_pos = int(episode["peak_pos"])
    end_pos = int(recovery_pos) if recovery_pos is not None else len(net_value) - 1
    episode["depth"] = _round_float(trough_value / peak_value - 1.0) if peak_value > 0 else 0.0
    episode["recovery_pos"] = recovery_pos
    episode["recovery_at"] = equity_point_label(data, recovery_pos) if recovery_pos is not None else ""
    underwater_end = int(recovery_pos) - 1 if recovery_pos is not None else end_pos
    episode["underwater_bars"] = _bar_span_length(data, peak_pos + 1, underwater_end)
    episode["recovery_bars"] = _bar_span_length(data, peak_pos + 1, end_pos)


def _public_drawdown_episode_row(episode: dict[str, object], data: pd.DataFrame) -> dict[str, object]:
    peak_pos = int(episode["peak_pos"])
    trough_pos = int(episode["trough_pos"])
    return {
        "episode_no": int(episode["episode_no"]),
        "start_at": equity_point_label(data, peak_pos),
        "trough_at": equity_point_label(data, trough_pos),
        "recovery_at": str(episode["recovery_at"]),
        "peak_net_value": _round_float(float(episode["peak_net_value"])),
        "trough_net_value": _round_float(float(episode["trough_net_value"])),
        "depth": _round_float(float(episode["depth"])),
        "underwater_bars": int(episode["underwater_bars"]),
        "recovery_bars": int(episode["recovery_bars"]),
        "recovered": episode["recovery_pos"] is not None,
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


def _max_drawdown_bar_duration(data: pd.DataFrame, equity: pd.Series) -> int:
    peak = -math.inf
    underwater_start: int | None = None
    best = 0
    for pos, value in enumerate(pd.to_numeric(equity, errors="coerce").tolist()):
        if pd.isna(value):
            continue
        if float(value) >= peak:
            if underwater_start is not None:
                best = max(best, _bar_span_length(data, underwater_start, pos - 1))
                underwater_start = None
            peak = float(value)
            continue
        if underwater_start is None:
            underwater_start = pos
        best = max(best, _bar_span_length(data, underwater_start, pos))
    return best


def _trailing_underwater_bar_length(data: pd.DataFrame, drawdown: pd.Series) -> int:
    values = pd.to_numeric(drawdown, errors="coerce").tolist()
    end_pos: int | None = None
    start_pos: int | None = None
    for pos in range(len(values) - 1, -1, -1):
        value = values[pos]
        if pd.isna(value) or float(value) >= 0:
            break
        end_pos = pos if end_pos is None else end_pos
        start_pos = pos
    if start_pos is None or end_pos is None:
        return 0
    return _bar_span_length(data, start_pos, end_pos)


def _bar_underwater_ratio(data: pd.DataFrame, drawdown: pd.Series) -> float:
    values = pd.to_numeric(drawdown, errors="coerce")
    if values.empty:
        return 0.0
    bar_states: dict[object, bool] = {}
    for pos, value in enumerate(values.tolist()):
        if pd.isna(value):
            continue
        key = _bar_identity(data, pos)
        bar_states[key] = bool(bar_states.get(key, False) or float(value) < 0)
    if not bar_states:
        return 0.0
    return _round_float(sum(bar_states.values()) / len(bar_states))


def _bar_worst_drawdown_series(data: pd.DataFrame, drawdown: pd.Series) -> pd.Series:
    """每根 K 只保留最深回撤，避免路径拆点后重复加权平均水下压力。"""
    values = pd.to_numeric(drawdown, errors="coerce")
    if values.empty:
        return pd.Series(dtype=float)
    bar_worst: dict[object, float] = {}
    for pos, value in enumerate(values.tolist()):
        if pd.isna(value):
            continue
        key = _bar_identity(data, pos)
        numeric = float(value)
        bar_worst[key] = min(float(bar_worst.get(key, numeric)), numeric)
    return pd.Series(bar_worst.values(), dtype=float)


def _bar_span_length(data: pd.DataFrame, start_pos: int, end_pos: int) -> int:
    if end_pos < start_pos:
        return 0
    labels = [_bar_identity(data, pos) for pos in range(start_pos, end_pos + 1)]
    return len(dict.fromkeys(labels))


def _bar_identity(data: pd.DataFrame, position: int) -> object:
    if position < 0 or position >= len(data):
        return position
    row = data.iloc[position]
    if "date" in data.columns:
        timestamp = pd.to_datetime(row["date"], errors="coerce")
        if pd.notna(timestamp):
            return ("date", pd.Timestamp(timestamp))
    if "trade_no" in data.columns and pd.notna(row["trade_no"]):
        return ("trade_no", _compact_numeric_label(row["trade_no"]))
    return ("position", position)


def _drawdown_value_series(data: pd.DataFrame, net_value: pd.Series) -> pd.Series:
    if "drawdown_net_value" not in data.columns:
        return net_value
    drawdown_value = pd.to_numeric(data["drawdown_net_value"], errors="coerce").reset_index(drop=True)
    return drawdown_value.fillna(net_value)


def _drawdown_path_row(row: dict[str, object], value: float, point_type: str) -> dict[str, object]:
    result = dict(row)
    result["_drawdown_path_value"] = float(value)
    result["_drawdown_point_type"] = point_type
    return result


def _optional_datetime_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([pd.NaT] * len(frame))
    return pd.to_datetime(frame[column], errors="coerce").reset_index(drop=True)


def _optional_numeric_column(frame: pd.DataFrame, column: str, *, fallback: object | None = None) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").reset_index(drop=True)
    if fallback is not None:
        return pd.to_numeric(fallback, errors="coerce").reset_index(drop=True)
    return pd.Series([pd.NA] * len(frame))


def _drawdown_path_needs_settlement_point(net: object, adverse: object) -> bool:
    if pd.isna(adverse):
        return True
    return not math.isclose(float(net), float(adverse), rel_tol=0.0, abs_tol=1e-12)


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
