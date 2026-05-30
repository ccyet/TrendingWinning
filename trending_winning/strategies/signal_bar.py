from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.data.schema import normalize_bars, normalize_symbol
from trending_winning.detectors.base import Detector, validate_detector_events
from trending_winning.strategies.base import ORDER_COLUMNS, empty_orders
from trending_winning.strategies.diagnostics import STRATEGY_FILTER_DECISION_COLUMNS, empty_strategy_filter_decisions

SUPPORTED_SIDE_MODES = ("both", "long_only", "short_only")
SIDE_MODE_ALLOWED_SIDES = {
    "both": frozenset({"long", "short"}),
    "long_only": frozenset({"long"}),
    "short_only": frozenset({"short"}),
}


@dataclass(frozen=True)
class SignalBarStopStrategyConfig:
    """信号 K 挂单策略参数；风险阈值随订单透传，由撮合层统一判定。"""

    name: str = "signal_bar_stop"
    risk_reward: float = 2.0
    max_holding_bars: int = 12
    max_actual_risk_pct: float | None = None
    max_chase_pct: float | None = None
    side_mode: str = "both"


class SignalBarStopStrategy:
    """信号 K 上下方挂单策略；只消费一个识别器输出的事件。"""

    def __init__(self, detector: Detector, config: SignalBarStopStrategyConfig | None = None) -> None:
        self.detector = detector
        self.config = config or SignalBarStopStrategyConfig()
        self.name = self.config.name
        self.last_filter_decisions = empty_strategy_filter_decisions()
        if self.config.risk_reward <= 0:
            raise ValueError("risk_reward 必须大于 0。")
        if self.config.max_holding_bars < 1:
            raise ValueError("max_holding_bars 至少需要 1。")
        if self.config.max_actual_risk_pct is not None and self.config.max_actual_risk_pct <= 0:
            raise ValueError("max_actual_risk_pct 必须大于 0 或设为 None。")
        if self.config.max_chase_pct is not None and self.config.max_chase_pct <= 0:
            raise ValueError("max_chase_pct 必须大于 0 或设为 None。")
        self.allowed_sides = _allowed_sides_for_mode(self.config.side_mode)

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        self.last_filter_decisions = empty_strategy_filter_decisions()
        events = validate_detector_events(
            self.detector.detect(bars, timeframe=timeframe),
            detector_name=str(getattr(self.detector, "name", "")),
        )
        if events.empty:
            return empty_orders()
        side = events["direction"].astype(str).str.strip().str.lower()
        tradable_mask = side.isin(["long", "short"])
        side_mode_mask = side.isin(self.allowed_sides)
        numeric = events.loc[:, ["entry_price", "stop_price", "signal_price"]].apply(pd.to_numeric, errors="coerce")
        signal_bar_index = pd.to_numeric(events["bar_index"], errors="coerce")
        valid = (
            tradable_mask
            & side_mode_mask
            & pd.to_datetime(events["date"], errors="coerce").notna()
            & signal_bar_index.notna()
            & signal_bar_index.ge(0)
            & events["stock_code"].map(normalize_symbol).ne("")
            & numeric["entry_price"].gt(0)
            & numeric["stop_price"].gt(0)
            & numeric["entry_price"].ne(numeric["stop_price"])
        )
        liquid = _signal_bar_liquidity_mask(bars, events)
        accepted_mask = valid & liquid
        self.last_filter_decisions = _signal_filter_decisions(
            events,
            accepted_mask,
            liquid,
            tradable_mask,
            side_mode_mask,
            strategy_name=self.name,
        )
        if not bool(accepted_mask.any()):
            return empty_orders()

        tradable = events.loc[accepted_mask].copy()
        numeric = numeric.loc[accepted_mask]
        side = side.loc[accepted_mask]
        signal_bar_index = signal_bar_index.loc[accepted_mask].astype(int)
        direction = side.map({"long": 1.0, "short": -1.0}).astype(float)
        risk_per_share = (numeric["entry_price"] - numeric["stop_price"]).abs()
        actual_risk_pct = risk_per_share / numeric["entry_price"]
        chase_pct = (numeric["entry_price"] - numeric["signal_price"]).abs() / numeric["signal_price"].where(
            numeric["signal_price"].gt(0)
        )
        chase_pct = chase_pct.fillna(0.0)
        target_price = numeric["entry_price"] + direction * risk_per_share * self.config.risk_reward

        result = pd.DataFrame(
            {
                "order_id": self.name + ":" + tradable["event_id"].astype(str),
                "strategy_name": self.name,
                "detector_name": tradable["detector_name"].to_numpy(),
                "event_id": tradable["event_id"].to_numpy(),
                "event_type": tradable["event_type"].to_numpy(),
                "stock_code": tradable["stock_code"].to_numpy(),
                "timeframe": tradable["timeframe"].to_numpy(),
                "signal_date": tradable["date"].to_numpy(),
                "signal_bar_index": signal_bar_index.to_numpy(),
                "side": side.to_numpy(),
                "signal_price": numeric["signal_price"].astype(float).to_numpy(),
                "entry_price": numeric["entry_price"].astype(float).to_numpy(),
                "stop_price": numeric["stop_price"].astype(float).to_numpy(),
                "target_price": target_price.astype(float).to_numpy(),
                "max_holding_bars": int(self.config.max_holding_bars),
                "max_actual_risk_pct": self.config.max_actual_risk_pct,
                "max_chase_pct": self.config.max_chase_pct,
                "metadata": [
                    {
                        **dict(metadata),
                        "actual_risk_pct": round(float(risk), 12),
                        "chase_pct": round(float(chase), 12),
                    }
                    for metadata, risk, chase in zip(
                        tradable["metadata"].to_list(),
                        actual_risk_pct.to_numpy(),
                        chase_pct.to_numpy(),
                        strict=True,
                    )
                ],
            },
            columns=ORDER_COLUMNS,
        )
        return result


def _signal_bar_liquidity_mask(bars: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """判断事件对应的信号 K 是否有真实成交；缺量额字段时保持兼容。"""
    if events.empty or bars.empty or "volume" not in bars.columns or "amount" not in bars.columns:
        return pd.Series([True] * len(events), index=events.index, dtype=bool)
    normalized = normalize_bars(bars)
    if normalized.empty:
        return pd.Series([True] * len(events), index=events.index, dtype=bool)
    signal_bars = normalized.copy()
    signal_bars["_bar_index"] = signal_bars.groupby("stock_code", sort=False).cumcount()
    signal_bars["_is_liquid_signal_bar"] = signal_bars["volume"].gt(0) & signal_bars["amount"].gt(0)
    lookup = signal_bars.set_index(["stock_code", "_bar_index"])["_is_liquid_signal_bar"]

    keys = pd.MultiIndex.from_arrays(
        [
            events["stock_code"].map(normalize_symbol),
            pd.to_numeric(events["bar_index"], errors="coerce").fillna(-1).astype(int),
        ],
        names=["stock_code", "_bar_index"],
    )
    return pd.Series(lookup.reindex(keys).fillna(True).to_numpy(dtype=bool), index=events.index)


def _signal_filter_decisions(
    events: pd.DataFrame,
    accepted_mask: pd.Series,
    liquid_mask: pd.Series,
    tradable_mask: pd.Series,
    side_mode_mask: pd.Series,
    *,
    strategy_name: str,
) -> pd.DataFrame:
    """记录信号层过滤结果；解释 detector 事件为什么没有变成订单。"""
    event_id = events["event_id"].fillna("").astype(str)
    accepted = accepted_mask.reindex(events.index, fill_value=False).to_numpy(dtype=bool)
    liquid = liquid_mask.reindex(events.index, fill_value=False).to_numpy(dtype=bool)
    tradable = tradable_mask.reindex(events.index, fill_value=False).to_numpy(dtype=bool)
    side_allowed = side_mode_mask.reindex(events.index, fill_value=False).to_numpy(dtype=bool)
    reason = np.select(
        [accepted, ~tradable, ~side_allowed, ~liquid],
        ["", "non_tradable_direction", "side_mode_filtered", "signal_bar_no_liquidity"],
        default="invalid_signal_order",
    )
    result = pd.DataFrame(
        {
            "order_id": strategy_name + ":" + event_id,
            "event_id": event_id,
            "strategy_name": strategy_name,
            "base_strategy_name": strategy_name,
            "detector_name": events["detector_name"].fillna("").astype(str).to_numpy(),
            "event_type": events["event_type"].fillna("").astype(str).to_numpy(),
            "stock_code": events["stock_code"].fillna("").astype(str).to_numpy(),
            "timeframe": events["timeframe"].fillna("").astype(str).to_numpy(),
            "signal_date": events["date"].to_numpy(),
            "signal_bar_index": pd.to_numeric(events["bar_index"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            "side": events["direction"].fillna("").astype(str).to_numpy(),
            "status": np.where(accepted, "accepted", "rejected"),
            "reason": reason,
            "filter_name": "signal_bar_adapter",
            "context_timeframe": "",
            "context_date": pd.NaT,
            "context_state": "",
        },
        columns=pd.Index(STRATEGY_FILTER_DECISION_COLUMNS),
    )
    return result


def _allowed_sides_for_mode(side_mode: str) -> frozenset[str]:
    """把界面/CLI 的方向模式翻译成订单方向集合。"""
    normalized = str(side_mode).strip().lower()
    if normalized not in SIDE_MODE_ALLOWED_SIDES:
        raise ValueError("side_mode 仅支持 both、long_only 或 short_only。")
    return SIDE_MODE_ALLOWED_SIDES[normalized]
