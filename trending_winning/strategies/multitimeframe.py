from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from trending_winning.data.schema import normalize_symbol
from trending_winning.strategies.base import ORDER_COLUMNS, Strategy, empty_orders
from trending_winning.strategies.diagnostics import (
    STRATEGY_FILTER_DECISION_COLUMNS,
    collect_strategy_filter_decisions,
    empty_strategy_filter_decisions,
)


@dataclass(frozen=True)
class TimeframeAlignmentConfig:
    """高周期方向门控参数；只描述上下文匹配规则，不绑定任何 detector。"""

    name: str = "higher_timeframe_aligned"
    context_timeframe: str = ""
    context_column: str = "direction"
    long_states: Sequence[str] = ("long", "bull", "up", "ai_long", "all_long")
    short_states: Sequence[str] = ("short", "bear", "down", "ai_short", "all_short")
    max_context_age: pd.Timedelta | str | None = None


class HigherTimeframeAlignmentStrategy:
    """高周期方向过滤策略；包装一个基础策略，只过滤订单，不改变识别器。"""

    def __init__(
        self,
        base_strategy: Strategy,
        higher_context: pd.DataFrame,
        config: TimeframeAlignmentConfig | None = None,
    ) -> None:
        self.base_strategy = base_strategy
        self.higher_context = higher_context.copy()
        self.config = config or TimeframeAlignmentConfig()
        self.name = self.config.name
        if not self.name:
            raise ValueError("name 不能为空。")
        if not self.config.context_column:
            raise ValueError("context_column 不能为空。")
        if self.config.max_context_age is not None and pd.Timedelta(self.config.max_context_age) < pd.Timedelta(0):
            raise ValueError("max_context_age 不能为负数。")
        self.last_filter_decisions = empty_strategy_filter_decisions()

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        self.last_filter_decisions = empty_strategy_filter_decisions()
        orders = self.base_strategy.generate_orders(bars, timeframe=timeframe)
        base_filter_decisions = collect_strategy_filter_decisions([self.base_strategy])
        if orders.empty:
            self.last_filter_decisions = base_filter_decisions
            return empty_orders()
        context = _normalize_context(self.higher_context, self.config.context_column)
        aligned = _merge_prior_context(orders, context)
        if aligned.empty:
            self.last_filter_decisions = base_filter_decisions
            return empty_orders()
        mask = self._alignment_mask(aligned)
        alignment_decisions = _filter_decision_frame(
            aligned,
            mask,
            strategy_name=self.name,
            base_strategy_name=self.base_strategy.name,
            context_timeframe=self.config.context_timeframe,
            long_states=self.config.long_states,
            short_states=self.config.short_states,
        )
        self.last_filter_decisions = _merge_filter_decision_frames(base_filter_decisions, alignment_decisions)
        accepted = aligned.loc[mask].copy()
        if accepted.empty:
            return empty_orders()

        accepted["strategy_name"] = self.name
        accepted_records = accepted.to_dict("records")
        accepted["metadata"] = [
            _metadata_with_context(record.get("metadata"), record, self.base_strategy.name, self.config.context_timeframe)
            for record in accepted_records
        ]
        return accepted[ORDER_COLUMNS].reset_index(drop=True)

    def _alignment_mask(self, frame: pd.DataFrame) -> pd.Series:
        state = frame["_context_state"].astype(str).str.strip().str.lower()
        side = frame["side"].astype(str).str.strip().str.lower()
        long_states = {str(item).strip().lower() for item in self.config.long_states}
        short_states = {str(item).strip().lower() for item in self.config.short_states}
        mask = (side.eq("long") & state.isin(long_states)) | (side.eq("short") & state.isin(short_states))
        if self.config.max_context_age is None:
            return mask
        max_age = pd.Timedelta(self.config.max_context_age)
        age = pd.to_datetime(frame["signal_date"]) - pd.to_datetime(frame["_context_date"])
        return mask & age.ge(pd.Timedelta(0)) & age.le(max_age)


def _normalize_context(context: pd.DataFrame, context_column: str) -> pd.DataFrame:
    required = {"date", "stock_code", context_column}
    missing = required.difference(context.columns)
    if missing:
        raise ValueError(f"高周期上下文缺少字段：{', '.join(sorted(missing))}")
    result = context.loc[:, ["date", "stock_code", context_column]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["stock_code"] = result["stock_code"].map(normalize_symbol).replace("", pd.NA)
    result["_context_state"] = result[context_column].astype(str).str.strip()
    result = result.dropna(subset=["date", "stock_code", "_context_state"])
    result = result.loc[result["_context_state"] != ""]
    return result[["date", "stock_code", "_context_state"]].sort_values(["stock_code", "date"]).reset_index(drop=True)


def _merge_prior_context(orders: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    left = orders.copy()
    left["_order_position"] = range(len(left))
    left["signal_date"] = pd.to_datetime(left["signal_date"], errors="coerce")
    left["stock_code"] = left["stock_code"].map(normalize_symbol).replace("", pd.NA)
    left["_invalid_order_key"] = left["signal_date"].isna() | left["stock_code"].isna()
    left["_context_date"] = pd.NaT
    left["_context_state"] = pd.NA
    if context.empty:
        return left.sort_values("_order_position").reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    valid = left.loc[~left["_invalid_order_key"]].copy()
    for symbol, group in valid.groupby("stock_code", sort=False):
        context_group = context.loc[context["stock_code"] == symbol, ["date", "_context_state"]]
        if context_group.empty:
            continue
        group_for_merge = group.drop(columns=["_context_date", "_context_state"])
        merged = pd.merge_asof(
            group_for_merge.sort_values("signal_date"),
            context_group.rename(columns={"date": "_context_date"}).sort_values("_context_date"),
            left_on="signal_date",
            right_on="_context_date",
            direction="backward",
            allow_exact_matches=True,
        )
        frames.append(merged[["_order_position", "_context_date", "_context_state"]])
    if not frames:
        return left.sort_values("_order_position").reset_index(drop=True)
    matched = pd.concat(frames, ignore_index=True).set_index("_order_position")
    result = left.set_index("_order_position")
    result.loc[matched.index, "_context_date"] = matched["_context_date"]
    result.loc[matched.index, "_context_state"] = matched["_context_state"]
    return result.reset_index().sort_values("_order_position").reset_index(drop=True)


def _filter_decision_frame(
    aligned: pd.DataFrame,
    accepted_mask: pd.Series,
    *,
    strategy_name: str,
    base_strategy_name: str,
    context_timeframe: str,
    long_states: Sequence[str],
    short_states: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row, accepted in zip(aligned.to_dict("records"), accepted_mask.to_numpy(dtype=bool), strict=True):
        rows.append(
            {
                "order_id": row.get("order_id", ""),
                "event_id": row.get("event_id", ""),
                "strategy_name": strategy_name,
                "base_strategy_name": base_strategy_name,
                "detector_name": row.get("detector_name", ""),
                "stock_code": row.get("stock_code", ""),
                "timeframe": row.get("timeframe", ""),
                "signal_date": row.get("signal_date", pd.NaT),
                "signal_bar_index": int(row.get("signal_bar_index", -1)),
                "side": row.get("side", ""),
                "status": "accepted" if bool(accepted) else "rejected",
                "reason": "" if bool(accepted) else _filter_reject_reason(row, long_states, short_states),
                "filter_name": "higher_timeframe_alignment",
                "context_timeframe": context_timeframe,
                "context_date": row.get("_context_date", pd.NaT),
                "context_state": row.get("_context_state", pd.NA),
            }
        )
    return pd.DataFrame(rows, columns=pd.Index(STRATEGY_FILTER_DECISION_COLUMNS))


def _merge_filter_decision_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """合并内外层策略过滤日志；组合回测据此看到完整拒绝链路。"""
    non_empty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty:
        return empty_strategy_filter_decisions()
    return pd.concat(non_empty, ignore_index=True)[STRATEGY_FILTER_DECISION_COLUMNS]


def _filter_reject_reason(row: Mapping[str, object], long_states: Sequence[str], short_states: Sequence[str]) -> str:
    if bool(row.get("_invalid_order_key", False)):
        return "invalid_order_key"
    if pd.isna(row.get("_context_date")) or pd.isna(row.get("_context_state")):
        return "higher_timeframe_no_context"
    state = str(row.get("_context_state", "")).strip().lower()
    side = str(row.get("side", "")).strip().lower()
    normalized_long_states = {str(item).strip().lower() for item in long_states}
    normalized_short_states = {str(item).strip().lower() for item in short_states}
    if side not in {"long", "short"}:
        return "higher_timeframe_mismatch"
    if side == "long" and state not in normalized_long_states:
        return "higher_timeframe_mismatch"
    if side == "short" and state not in normalized_short_states:
        return "higher_timeframe_mismatch"
    return "higher_timeframe_stale"


def _metadata_with_context(
    metadata: object,
    row: Mapping[str, object],
    base_strategy_name: str,
    context_timeframe: str,
) -> dict[str, object]:
    result = dict(metadata) if isinstance(metadata, dict) else {}
    result["base_strategy_name"] = base_strategy_name
    result["higher_timeframe"] = context_timeframe
    result["higher_context_date"] = pd.Timestamp(row["_context_date"])
    result["higher_state"] = str(row["_context_state"])
    return result
