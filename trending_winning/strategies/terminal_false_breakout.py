from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trending_winning.data.schema import CANONICAL_COLUMNS, normalize_bars, normalize_symbol
from trending_winning.strategies.base import ORDER_COLUMNS, Strategy, empty_orders
from trending_winning.strategies.diagnostics import STRATEGY_FILTER_DECISION_COLUMNS, empty_strategy_filter_decisions
from trending_winning.strategies.runtime import StrategyRunResult, execute_strategy


@dataclass(frozen=True)
class TerminalFalseBreakoutFilterConfig:
    """末端假突破过滤参数；默认关闭，只在用户启用后过滤开仓订单。"""

    enabled: bool = False
    detectors: tuple[str, ...] = ("trend", "channel")
    lookback: int = 40
    atr_period: int = 14
    min_regime_bars: int = 18
    extension_atr_multiple: float = 2.0
    edge_lookback: int = 8
    edge_pos: float = 0.90
    edge_min_count: int = 3
    weak_progress_atr: float = 0.35
    wick_ratio: float = 0.35
    min_score: int = 3


class TerminalFalseBreakoutFilterStrategy:
    """同级别末端假突破过滤器；包装基础策略，不修改 detector 或撮合逻辑。"""

    def __init__(self, base_strategy: Strategy, config: TerminalFalseBreakoutFilterConfig | None = None) -> None:
        self.base_strategy = base_strategy
        self.config = config or TerminalFalseBreakoutFilterConfig()
        self.name = str(getattr(base_strategy, "name", ""))
        self.last_filter_decisions = empty_strategy_filter_decisions()
        _validate_config(self.config)

    def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
        return self.generate_order_plan(bars, timeframe=timeframe).orders

    def generate_order_plan(self, bars: pd.DataFrame, *, timeframe: str = "") -> StrategyRunResult:
        base_run = execute_strategy(self.base_strategy, bars, timeframe=timeframe)
        if not self.config.enabled or base_run.orders.empty:
            self.last_filter_decisions = base_run.filter_decisions
            return StrategyRunResult(
                orders=base_run.orders,
                filter_decisions=self.last_filter_decisions,
                strategy_name=base_run.strategy_name or self.name,
            )

        scored = _score_orders(base_run.orders, bars, self.config)
        rejected_mask = scored["_terminal_reject"].to_numpy(dtype=bool)
        terminal_decisions = _terminal_filter_decisions(scored, ~rejected_mask)
        self.last_filter_decisions = _merge_filter_decisions(base_run.filter_decisions, terminal_decisions)

        accepted = scored.loc[~rejected_mask, ORDER_COLUMNS + ["_terminal_score", "_terminal_context"]].copy()
        if accepted.empty:
            return StrategyRunResult(
                orders=empty_orders(),
                filter_decisions=self.last_filter_decisions,
                strategy_name=base_run.strategy_name or self.name,
            )
        accepted["metadata"] = _metadata_with_terminal_state(
            accepted["metadata"],
            accepted["_terminal_score"],
            accepted["_terminal_context"],
        )
        return StrategyRunResult(
            orders=accepted.loc[:, ORDER_COLUMNS].reset_index(drop=True),
            filter_decisions=self.last_filter_decisions,
            strategy_name=base_run.strategy_name or self.name,
        )


def _validate_config(config: TerminalFalseBreakoutFilterConfig) -> None:
    if config.lookback < 3:
        raise ValueError("terminal_false_breakout lookback 至少需要 3。")
    if config.atr_period < 1:
        raise ValueError("terminal_false_breakout atr_period 至少需要 1。")
    if config.min_regime_bars < 1:
        raise ValueError("terminal_false_breakout min_regime_bars 至少需要 1。")
    if config.extension_atr_multiple < 0:
        raise ValueError("terminal_false_breakout extension_atr_multiple 不能为负数。")
    if config.edge_lookback < 1 or config.edge_min_count < 1:
        raise ValueError("terminal_false_breakout edge 参数至少需要 1。")
    if not 0 <= config.edge_pos <= 1:
        raise ValueError("terminal_false_breakout edge_pos 必须在 0 到 1 之间。")
    if config.weak_progress_atr < 0 or not 0 <= config.wick_ratio <= 1:
        raise ValueError("terminal_false_breakout 突破推进和影线参数非法。")
    if config.min_score < 1:
        raise ValueError("terminal_false_breakout min_score 至少需要 1。")


def _score_orders(
    orders: pd.DataFrame,
    bars: pd.DataFrame,
    config: TerminalFalseBreakoutFilterConfig,
) -> pd.DataFrame:
    features = _terminal_feature_frame(bars, config)
    result = orders.copy()
    result["_order_position"] = range(len(result))
    result["stock_code"] = result["stock_code"].map(normalize_symbol)
    result["signal_bar_index"] = pd.to_numeric(result["signal_bar_index"], errors="coerce").fillna(-1).astype(int)
    if features.empty:
        return _orders_with_empty_terminal_score(result)

    merged = result.merge(
        features,
        how="left",
        left_on=["stock_code", "signal_bar_index"],
        right_on=["stock_code", "_bar_index"],
        sort=False,
    ).sort_values("_order_position", kind="mergesort")
    side = merged["side"].fillna("").astype(str).str.strip().str.lower()
    detector = merged["detector_name"].fillna("").astype(str).str.strip()
    allowed_detector = detector.isin({str(item) for item in config.detectors})
    long_score = _long_terminal_score(merged, config)
    short_score = _short_terminal_score(merged, config)
    score = np.select([side.eq("long"), side.eq("short")], [long_score, short_score], default=0).astype(int)
    regime_ready = np.select(
        [side.eq("long"), side.eq("short")],
        [
            pd.to_numeric(merged["_up_regime_run"], errors="coerce").ge(config.min_regime_bars),
            pd.to_numeric(merged["_down_regime_run"], errors="coerce").ge(config.min_regime_bars),
        ],
        default=False,
    ).astype(bool)
    context = _terminal_context(score, merged)
    merged["_terminal_score"] = score
    merged["_terminal_context"] = context
    merged["_terminal_reject"] = allowed_detector & side.isin(["long", "short"]) & regime_ready & (score >= int(config.min_score))
    return merged


def _orders_with_empty_terminal_score(orders: pd.DataFrame) -> pd.DataFrame:
    result = orders.copy()
    result["_terminal_score"] = 0
    result["_terminal_context"] = "terminal_score=0"
    result["_terminal_reject"] = False
    return result


def _terminal_feature_frame(bars: pd.DataFrame, config: TerminalFalseBreakoutFilterConfig) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame()
    normalized = bars.loc[:, CANONICAL_COLUMNS].copy() if _looks_like_normalized_bars(bars) else normalize_bars(bars)
    if normalized.empty:
        return pd.DataFrame()
    frames = [_feature_group(group.reset_index(drop=True), config) for _, group in normalized.groupby("stock_code", sort=False)]
    return pd.concat(frames, ignore_index=True)


def _feature_group(group: pd.DataFrame, config: TerminalFalseBreakoutFilterConfig) -> pd.DataFrame:
    result = group.loc[:, ["date", "stock_code", "open", "high", "low", "close"]].copy()
    result["_bar_index"] = np.arange(len(result), dtype=int)
    open_ = pd.to_numeric(result["open"], errors="coerce").astype(float)
    high = pd.to_numeric(result["high"], errors="coerce").astype(float)
    low = pd.to_numeric(result["low"], errors="coerce").astype(float)
    close = pd.to_numeric(result["close"], errors="coerce").astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(config.atr_period, min_periods=1).mean()
    upper = high.rolling(config.lookback, min_periods=config.lookback).max().shift(1)
    lower = low.rolling(config.lookback, min_periods=config.lookback).min().shift(1)
    mid = (upper + lower) / 2.0
    width = (upper - lower).replace(0.0, np.nan)
    pos = (close - lower) / width
    up_regime = mid.gt(mid.shift(1))
    down_regime = mid.lt(mid.shift(1))
    prior_high = high.shift(1).rolling(config.edge_lookback, min_periods=1).max()
    prior_low = low.shift(1).rolling(config.edge_lookback, min_periods=1).min()
    candle_range = (high - low).replace(0.0, np.nan)

    result["_atr"] = atr
    result["_channel_mid"] = mid
    result["_channel_pos"] = pos
    result["_up_regime_run"] = _consecutive_true_count(up_regime.to_numpy(dtype=bool))
    result["_down_regime_run"] = _consecutive_true_count(down_regime.to_numpy(dtype=bool))
    result["_upper_edge_count"] = pos.ge(config.edge_pos).rolling(config.edge_lookback, min_periods=1).sum()
    result["_lower_edge_count"] = pos.le(1.0 - config.edge_pos).rolling(config.edge_lookback, min_periods=1).sum()
    result["_long_progress_atr"] = ((high - prior_high).clip(lower=0.0) / atr).replace([np.inf, -np.inf], np.nan)
    result["_short_progress_atr"] = ((prior_low - low).clip(lower=0.0) / atr).replace([np.inf, -np.inf], np.nan)
    result["_extension_atr"] = ((close - mid).abs() / atr).replace([np.inf, -np.inf], np.nan)
    result["_upper_wick_ratio"] = ((high - np.maximum(open_, close)) / candle_range).replace([np.inf, -np.inf], np.nan)
    result["_lower_wick_ratio"] = ((np.minimum(open_, close) - low) / candle_range).replace([np.inf, -np.inf], np.nan)
    return result


def _consecutive_true_count(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=int)
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        out[index] = current
    return out


def _long_terminal_score(frame: pd.DataFrame, config: TerminalFalseBreakoutFilterConfig) -> np.ndarray:
    conditions = [
        pd.to_numeric(frame["_up_regime_run"], errors="coerce").ge(config.min_regime_bars),
        pd.to_numeric(frame["_extension_atr"], errors="coerce").ge(config.extension_atr_multiple),
        pd.to_numeric(frame["_upper_edge_count"], errors="coerce").ge(config.edge_min_count),
        pd.to_numeric(frame["_long_progress_atr"], errors="coerce").le(config.weak_progress_atr),
        pd.to_numeric(frame["_upper_wick_ratio"], errors="coerce").ge(config.wick_ratio),
    ]
    return np.vstack([condition.fillna(False).to_numpy(dtype=bool) for condition in conditions]).sum(axis=0)


def _short_terminal_score(frame: pd.DataFrame, config: TerminalFalseBreakoutFilterConfig) -> np.ndarray:
    conditions = [
        pd.to_numeric(frame["_down_regime_run"], errors="coerce").ge(config.min_regime_bars),
        pd.to_numeric(frame["_extension_atr"], errors="coerce").ge(config.extension_atr_multiple),
        pd.to_numeric(frame["_lower_edge_count"], errors="coerce").ge(config.edge_min_count),
        pd.to_numeric(frame["_short_progress_atr"], errors="coerce").le(config.weak_progress_atr),
        pd.to_numeric(frame["_lower_wick_ratio"], errors="coerce").ge(config.wick_ratio),
    ]
    return np.vstack([condition.fillna(False).to_numpy(dtype=bool) for condition in conditions]).sum(axis=0)


def _terminal_context(score: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [
            (
                f"terminal_score={int(item_score)};"
                f"regime_run={int(max(up_run, down_run))};"
                f"extension_atr={float(extension):.4f};"
                f"channel_pos={float(channel_pos):.4f}"
            )
            for item_score, up_run, down_run, extension, channel_pos in zip(
                score,
                pd.to_numeric(frame["_up_regime_run"], errors="coerce").fillna(0),
                pd.to_numeric(frame["_down_regime_run"], errors="coerce").fillna(0),
                pd.to_numeric(frame["_extension_atr"], errors="coerce").fillna(0.0),
                pd.to_numeric(frame["_channel_pos"], errors="coerce").fillna(0.0),
                strict=True,
            )
        ],
        dtype=object,
    )


def _terminal_filter_decisions(scored: pd.DataFrame, accepted_mask: np.ndarray) -> pd.DataFrame:
    accepted = np.asarray(accepted_mask, dtype=bool)
    return pd.DataFrame(
        {
            "order_id": scored["order_id"].fillna("").astype(str).to_numpy(),
            "event_id": scored["event_id"].fillna("").astype(str).to_numpy(),
            "strategy_name": scored["strategy_name"].fillna("").astype(str).to_numpy(),
            "base_strategy_name": scored["strategy_name"].fillna("").astype(str).to_numpy(),
            "detector_name": scored["detector_name"].fillna("").astype(str).to_numpy(),
            "event_type": scored["event_type"].fillna("").astype(str).to_numpy(),
            "stock_code": scored["stock_code"].fillna("").astype(str).to_numpy(),
            "timeframe": scored["timeframe"].fillna("").astype(str).to_numpy(),
            "signal_date": scored["signal_date"].to_numpy(),
            "signal_bar_index": pd.to_numeric(scored["signal_bar_index"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            "side": scored["side"].fillna("").astype(str).to_numpy(),
            "status": np.where(accepted, "accepted", "rejected"),
            "reason": np.where(accepted, "", "terminal_false_breakout_risk"),
            "filter_name": "terminal_false_breakout_filter",
            "context_timeframe": scored["timeframe"].fillna("").astype(str).to_numpy(),
            "context_date": scored["signal_date"].to_numpy(),
            "context_state": scored["_terminal_context"].astype(str).to_numpy(),
        },
        columns=pd.Index(STRATEGY_FILTER_DECISION_COLUMNS),
    )


def _metadata_with_terminal_state(
    metadata: pd.Series,
    terminal_score: pd.Series,
    terminal_context: pd.Series,
) -> list[dict[str, object]]:
    return [
        {
            **(dict(item) if isinstance(item, dict) else {}),
            "terminal_false_breakout_score": int(score),
            "terminal_false_breakout_context": str(context),
        }
        for item, score, context in zip(metadata, terminal_score, terminal_context, strict=True)
    ]


def _merge_filter_decisions(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty:
        return empty_strategy_filter_decisions()
    return pd.concat(non_empty, ignore_index=True)[STRATEGY_FILTER_DECISION_COLUMNS]


def _looks_like_normalized_bars(bars: pd.DataFrame) -> bool:
    return set(CANONICAL_COLUMNS).issubset(bars.columns) and pd.api.types.is_datetime64_any_dtype(bars["date"])
