from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SinglePositionGateDecision:
    """单策略满仓门控结果；accepted 表示候选成交可进入最终交易表。"""

    candidate: Mapping[str, object]
    status: str
    reason: str = ""


def apply_single_position_gate(candidates: Sequence[Mapping[str, object]]) -> list[SinglePositionGateDecision]:
    """按真实入场时间做单策略满仓门控，一笔未平仓前不允许开第二笔。"""
    decisions: list[SinglePositionGateDecision] = []
    open_until_date: pd.Timestamp | None = None
    for candidate in sorted(candidates, key=single_position_candidate_sort_key):
        trade = candidate.get("trade")
        if not isinstance(trade, Mapping):
            decisions.append(SinglePositionGateDecision(candidate=candidate, status="rejected", reason="invalid_trade"))
            continue
        entry_date = pd.to_datetime(trade.get("entry_date", pd.NaT), errors="coerce")
        if open_until_date is not None and pd.notna(entry_date) and entry_date <= open_until_date:
            decisions.append(SinglePositionGateDecision(candidate=candidate, status="rejected", reason="already_open"))
            continue
        decisions.append(SinglePositionGateDecision(candidate=candidate, status="accepted"))
        exit_date = pd.to_datetime(trade.get("exit_date", pd.NaT), errors="coerce")
        open_until_date = pd.Timestamp(exit_date) if pd.notna(exit_date) else None
    return decisions


def single_position_candidate_sort_key(candidate: Mapping[str, object]) -> tuple[bool, pd.Timestamp, int, str, str]:
    """单策略满仓门控按真实入场时间排序；信号时间只负责生成候选订单。"""
    trade = candidate.get("trade")
    if not isinstance(trade, Mapping):
        return (True, pd.Timestamp.max, int(candidate.get("order_sequence", 0)), "", "")
    entry_date = pd.to_datetime(trade.get("entry_date", pd.NaT), errors="coerce")
    missing_entry = bool(pd.isna(entry_date))
    entry_key = pd.Timestamp.max if missing_entry else pd.Timestamp(entry_date)
    return (
        missing_entry,
        entry_key,
        int(candidate.get("order_sequence", 0)),
        str(trade.get("stock_code", "")),
        str(trade.get("order_id", "")),
    )
