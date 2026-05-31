from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from trending_winning.backtest.stats import STAT_KEYS, compute_grouped_trade_statistics


SETUP_STAT_FIELDS = ("detector_name", "event_type", "side")
SIGNAL_LIFECYCLE_FIELDS = ("detector_name", "event_type", "side", "exit_reason")
TRADE_PATH_DISTRIBUTION_COLUMNS = pd.Index(
    [
        "dimension",
        "bucket",
        "bucket_order",
        "trade_count",
        "win_rate",
        "avg_return",
        "avg_r_multiple",
        "avg_mae_r",
        "avg_mfe_r",
        "avg_holding_bars",
    ]
)


def grouped_trade_statistics(trades: pd.DataFrame, *, by: str | Sequence[str]) -> pd.DataFrame:
    """按指定字段汇总逐笔交易；缺字段时返回稳定空表。"""
    fields = (by,) if isinstance(by, str) else tuple(by)
    missing = [field for field in fields if field not in trades.columns]
    if missing:
        return pd.DataFrame(columns=pd.Index([*fields, *STAT_KEYS]))
    return compute_grouped_trade_statistics(trades, by=by)


def strategy_trade_statistics(
    trades: pd.DataFrame,
    strategies: Sequence[object],
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按策略汇总成交表现；保留已启用但没有成交的策略。"""
    columns = pd.Index(["strategy_name", *STAT_KEYS])
    stats = grouped_trade_statistics(trades, by="strategy_name").reindex(columns=columns)
    strategy_names = strategy_names_for_statistics(strategies, order_decisions, filter_decisions)
    if not strategy_names:
        return stats
    existing_names = set()
    if not stats.empty and "strategy_name" in stats.columns:
        existing_names = {name for name in stats["strategy_name"].map(_label) if name}
    missing_names = [name for name in strategy_names if name not in existing_names]
    if not missing_names:
        return _sort_strategy_statistics(stats, strategy_names)
    zero_rows = pd.DataFrame(
        [{"strategy_name": strategy_name, **{stat_key: 0.0 for stat_key in STAT_KEYS}} for strategy_name in missing_names],
        columns=columns,
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_strategy_statistics(pd.concat(frames, ignore_index=True), strategy_names)


def detector_trade_statistics(
    trades: pd.DataFrame,
    detector_names: Sequence[object],
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按识别模块汇总成交表现；保留已启用但没有成交的 detector。"""
    columns = pd.Index(["detector_name", *STAT_KEYS])
    stats = grouped_trade_statistics(trades, by="detector_name").reindex(columns=columns)
    detectors = _detector_names_for_statistics(detector_names, order_decisions, filter_decisions)
    if not detectors:
        return stats
    existing_names = set()
    if not stats.empty and "detector_name" in stats.columns:
        existing_names = {name for name in stats["detector_name"].map(_label) if name}
    missing_names = [name for name in detectors if name not in existing_names]
    if not missing_names:
        return _sort_detector_statistics(stats, detectors)
    zero_rows = pd.DataFrame(
        [{"detector_name": detector_name, **{stat_key: 0.0 for stat_key in STAT_KEYS}} for detector_name in missing_names],
        columns=columns,
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_detector_statistics(pd.concat(frames, ignore_index=True), detectors)


def setup_trade_statistics(
    trades: pd.DataFrame,
    order_decisions: pd.DataFrame | None = None,
    filter_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 setup 汇总成交表现；保留只有信号或拒单、没有成交的 setup。"""
    columns = pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS])
    stats = grouped_trade_statistics(trades, by=SETUP_STAT_FIELDS).reindex(columns=columns)
    setup_keys = _setup_keys_from_decisions(order_decisions, filter_decisions)
    if setup_keys.empty:
        return stats
    existing_keys = set(_setup_key_tuples(stats))
    missing_keys = [
        tuple(row)
        for row in setup_keys.loc[:, SETUP_STAT_FIELDS].itertuples(index=False, name=None)
        if tuple(row) not in existing_keys
    ]
    if not missing_keys:
        return _sort_setup_statistics(stats)
    zero_rows = pd.DataFrame(
        [{**dict(zip(SETUP_STAT_FIELDS, key, strict=True)), **{stat_key: 0.0 for stat_key in STAT_KEYS}} for key in missing_keys],
        columns=columns,
    )
    frames = [frame for frame in (stats, zero_rows) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    return _sort_setup_statistics(pd.concat(frames, ignore_index=True))


def signal_lifecycle_statistics(trades: pd.DataFrame) -> pd.DataFrame:
    """按信号形态、方向和退出原因汇总绩效，用于观察开仓信号到平仓结果的完整路径。"""
    return grouped_trade_statistics(trades, by=SIGNAL_LIFECYCLE_FIELDS)


def trade_path_distribution_statistics(trades: pd.DataFrame) -> pd.DataFrame:
    """按持仓周期和风险路径分桶汇总成交质量，帮助定位策略赚亏在哪里发生。"""
    if trades.empty:
        return pd.DataFrame(columns=TRADE_PATH_DISTRIBUTION_COLUMNS)

    frame = trades.copy()
    frame["_return_decimal"] = pd.to_numeric(frame.get("return_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) / 100.0
    frame["_holding_bars"] = pd.to_numeric(frame.get("holding_bars", pd.Series(dtype=float)), errors="coerce")
    frame["_r_multiple"] = pd.to_numeric(frame.get("r_multiple", pd.Series(dtype=float)), errors="coerce")
    frame["_mae_r"] = pd.to_numeric(frame.get("mae_r", pd.Series(dtype=float)), errors="coerce")
    frame["_mfe_r"] = pd.to_numeric(frame.get("mfe_r", pd.Series(dtype=float)), errors="coerce")

    rows: list[dict[str, object]] = []
    rows.extend(_bucket_distribution_rows(frame, "_holding_bars", "持有K数", _holding_bar_bucket))
    rows.extend(_bucket_distribution_rows(frame, "_r_multiple", "R倍数", _r_multiple_bucket))
    rows.extend(_bucket_distribution_rows(frame, "_mae_r", "最大不利R", _mae_r_bucket))
    rows.extend(_bucket_distribution_rows(frame, "_mfe_r", "最大有利R", _mfe_r_bucket))
    if not rows:
        return pd.DataFrame(columns=TRADE_PATH_DISTRIBUTION_COLUMNS)
    return pd.DataFrame(rows).sort_values(["dimension", "bucket_order"], kind="mergesort").reset_index(drop=True)[
        TRADE_PATH_DISTRIBUTION_COLUMNS
    ]


def trade_dated_equity_curve(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """给单策略成交净值补 exit_date，避免月度统计落到纯 trade_no 轴。"""
    equity = equity_curve.copy()
    if "date" in equity.columns:
        return equity
    trade_frame = trades.copy()
    if equity.empty or trade_frame.empty or "exit_date" not in trade_frame.columns:
        return equity
    if "trade_no" not in equity.columns:
        return equity
    dated = equity.merge(
        trade_frame.assign(trade_no=range(1, len(trade_frame) + 1))[["trade_no", "exit_date"]],
        on="trade_no",
        how="left",
    )
    if 0 in set(pd.to_numeric(dated["trade_no"], errors="coerce").dropna().astype(int)):
        first_exit_date = pd.to_datetime(trade_frame["exit_date"], errors="coerce").dropna().min()
        if pd.notna(first_exit_date):
            dated.loc[dated["trade_no"].eq(0), "exit_date"] = first_exit_date
    return dated.rename(columns={"exit_date": "date"})


def _bucket_distribution_rows(
    frame: pd.DataFrame,
    value_column: str,
    dimension: str,
    bucket_fn,
) -> list[dict[str, object]]:
    if value_column not in frame.columns:
        return []
    bucketed = frame.loc[frame[value_column].notna()].copy()
    if bucketed.empty:
        return []
    bucket_values = bucketed[value_column].map(bucket_fn)
    bucketed["_bucket"] = [bucket for bucket, _order in bucket_values]
    bucketed["_bucket_order"] = [order for _bucket, order in bucket_values]
    rows: list[dict[str, object]] = []
    for (bucket, bucket_order), group in bucketed.groupby(["_bucket", "_bucket_order"], sort=True):
        returns = pd.to_numeric(group["_return_decimal"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "dimension": dimension,
                "bucket": str(bucket),
                "bucket_order": int(bucket_order),
                "trade_count": float(len(group)),
                "win_rate": _mean_or_zero(returns.gt(0).astype(float)),
                "avg_return": _mean_or_zero(returns),
                "avg_r_multiple": _mean_or_zero(group["_r_multiple"]),
                "avg_mae_r": _mean_or_zero(group["_mae_r"]),
                "avg_mfe_r": _mean_or_zero(group["_mfe_r"]),
                "avg_holding_bars": _mean_or_zero(group["_holding_bars"]),
            }
        )
    return rows


def _holding_bar_bucket(value: float) -> tuple[str, int]:
    if value <= 1:
        return "1K", 0
    if value <= 3:
        return "2-3K", 1
    if value <= 8:
        return "4-8K", 2
    if value <= 16:
        return "9-16K", 3
    return "17K+", 4


def _r_multiple_bucket(value: float) -> tuple[str, int]:
    if value <= -1:
        return "<=-1R", 0
    if value < 0:
        return "-1R~0R", 1
    if value < 1:
        return "0R~1R", 2
    if value < 2:
        return "1R~2R", 3
    return ">=2R", 4


def _mae_r_bucket(value: float) -> tuple[str, int]:
    if value <= -1:
        return "<=-1R", 0
    if value <= -0.5:
        return "-1R~-0.5R", 1
    if value < 0:
        return "-0.5R~0R", 2
    return "0R", 3


def _mfe_r_bucket(value: float) -> tuple[str, int]:
    if value <= 0:
        return "0R", 0
    if value < 0.5:
        return "0R~0.5R", 1
    if value < 1:
        return "0.5R~1R", 2
    if value < 2:
        return "1R~2R", 3
    return ">=2R", 4


def _mean_or_zero(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return 0.0
    return float(round(float(numeric.mean()), 12))


def strategy_names_for_statistics(
    strategies: Sequence[object],
    *decision_frames: pd.DataFrame | None,
) -> tuple[str, ...]:
    """生成策略统计行的稳定顺序，优先使用本次实际执行的策略对象。"""
    names: list[str] = []
    for strategy in strategies:
        name = _label(strategy if isinstance(strategy, str) else getattr(strategy, "name", ""))
        if name and name not in names:
            names.append(name)
    for strategy_name in _names_from_decisions("strategy_name", *decision_frames):
        if strategy_name not in names:
            names.append(strategy_name)
    return tuple(names)


def _detector_names_for_statistics(
    detector_names: Sequence[object],
    *decision_frames: pd.DataFrame | None,
) -> tuple[str, ...]:
    names: list[str] = []
    for detector_name in detector_names:
        name = _label(detector_name)
        if name and name not in names:
            names.append(name)
    for detector_name in _names_from_decisions("detector_name", *decision_frames):
        if detector_name not in names:
            names.append(detector_name)
    return tuple(names)


def _names_from_decisions(field: str, *decision_frames: pd.DataFrame | None) -> tuple[str, ...]:
    names: list[str] = []
    for frame in decision_frames:
        if frame is None or frame.empty or field not in frame.columns:
            continue
        for name in frame[field].map(_label):
            if name and name not in names:
                names.append(name)
    return tuple(names)


def _sort_strategy_statistics(stats: pd.DataFrame, strategy_names: Sequence[str]) -> pd.DataFrame:
    columns = pd.Index(["strategy_name", *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    order = {name: index for index, name in enumerate(strategy_names)}
    result = stats.reindex(columns=columns).copy()
    result["_strategy_label"] = result["strategy_name"].map(_label)
    result["_strategy_order"] = result["_strategy_label"].map(lambda name: order.get(name, len(order)))
    return (
        result.sort_values(["_strategy_order", "_strategy_label"], kind="mergesort")
        .drop(columns=["_strategy_order", "_strategy_label"])
        .reset_index(drop=True)
    )


def _sort_detector_statistics(stats: pd.DataFrame, detector_names: Sequence[str]) -> pd.DataFrame:
    columns = pd.Index(["detector_name", *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    order = {name: index for index, name in enumerate(detector_names)}
    result = stats.reindex(columns=columns).copy()
    result["_detector_label"] = result["detector_name"].map(_label)
    result["_detector_order"] = result["_detector_label"].map(lambda name: order.get(name, len(order)))
    return (
        result.sort_values(["_detector_order", "_detector_label"], kind="mergesort")
        .drop(columns=["_detector_order", "_detector_label"])
        .reset_index(drop=True)
    )


def _setup_keys_from_decisions(*decision_frames: pd.DataFrame | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frame in decision_frames:
        if frame is None or frame.empty or not set(SETUP_STAT_FIELDS).issubset(frame.columns):
            continue
        setup = frame.loc[:, SETUP_STAT_FIELDS].copy()
        for setup_field in SETUP_STAT_FIELDS:
            setup[setup_field] = setup[setup_field].map(_label)
        present = setup.loc[:, SETUP_STAT_FIELDS].ne("").all(axis=1)
        if bool(present.any()):
            frames.append(setup.loc[present, SETUP_STAT_FIELDS])
    if not frames:
        return pd.DataFrame(columns=pd.Index(SETUP_STAT_FIELDS))
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .sort_values(list(SETUP_STAT_FIELDS), kind="mergesort")
        .reset_index(drop=True)
    )


def _setup_key_tuples(stats: pd.DataFrame) -> list[tuple[str, str, str]]:
    if stats.empty or not set(SETUP_STAT_FIELDS).issubset(stats.columns):
        return []
    normalized = stats.loc[:, SETUP_STAT_FIELDS].copy()
    for setup_field in SETUP_STAT_FIELDS:
        normalized[setup_field] = normalized[setup_field].map(_label)
    return [tuple(row) for row in normalized.itertuples(index=False, name=None)]


def _sort_setup_statistics(stats: pd.DataFrame) -> pd.DataFrame:
    columns = pd.Index([*SETUP_STAT_FIELDS, *STAT_KEYS])
    if stats.empty:
        return pd.DataFrame(columns=columns)
    return (
        stats.reindex(columns=columns)
        .sort_values(list(SETUP_STAT_FIELDS), kind="mergesort")
        .reset_index(drop=True)
    )


def _label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
