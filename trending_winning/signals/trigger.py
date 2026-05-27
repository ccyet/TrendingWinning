from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TriggerConfig:
    """突破触发参数；控制突破缓冲、量能确认和是否要求标志 K。"""

    close_buffer_pct: float = 0.0
    volume_multiple: float = 1.5
    volume_lookback: int = 20
    require_landmark: bool = True


def detect_breakout_triggers(bars: pd.DataFrame, config: TriggerConfig | None = None) -> pd.DataFrame:
    cfg = config or TriggerConfig()
    if cfg.close_buffer_pct < 0:
        raise ValueError("close_buffer_pct 不能为负数。")
    if cfg.volume_multiple <= 0:
        raise ValueError("volume_multiple 必须大于 0。")
    if cfg.volume_lookback < 2:
        raise ValueError("volume_lookback 至少需要 2。")

    result = bars.sort_values(["stock_code", "date"]).reset_index(drop=True).copy()
    grouped = result.groupby("stock_code", sort=False)
    result["_prev_upper"] = grouped["channel_upper"].shift(1)
    result["_prev_direction"] = grouped["channel_direction"].shift(1)
    result["_volume_mean"] = grouped["volume"].transform(
        lambda series: pd.to_numeric(series, errors="coerce").rolling(cfg.volume_lookback, min_periods=2).mean().shift(1)
    )

    price_pass = pd.to_numeric(result["close"], errors="coerce") > result["_prev_upper"] * (1.0 + cfg.close_buffer_pct)
    trend_pass = result["_prev_direction"].isin(["up", "flat"])
    volume_pass = pd.to_numeric(result["volume"], errors="coerce") >= result["_volume_mean"] * cfg.volume_multiple
    if cfg.require_landmark:
        landmark_pass = result.get("is_landmark", False)
    else:
        landmark_pass = True

    result["breakout_trigger"] = (price_pass & trend_pass & volume_pass & landmark_pass).fillna(False)
    result["breakout_level"] = result["_prev_upper"]
    result["trigger_price"] = pd.NA
    result.loc[result["breakout_trigger"], "trigger_price"] = result.loc[result["breakout_trigger"], "close"]
    return result.drop(columns=["_prev_upper", "_prev_direction", "_volume_mean"])
