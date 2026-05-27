from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LandmarkConfig:
    """标志 K 参数；用振幅、成交量和实体比例筛选关键 K 线。"""

    lookback: int = 20
    range_multiple: float = 1.8
    volume_multiple: float = 1.8
    min_body_ratio: float = 0.5


def detect_landmark_candles(bars: pd.DataFrame, config: LandmarkConfig | None = None) -> pd.DataFrame:
    cfg = config or LandmarkConfig()
    if cfg.lookback < 2:
        raise ValueError("lookback 至少需要 2。")
    if cfg.range_multiple <= 0 or cfg.volume_multiple <= 0:
        raise ValueError("range_multiple 和 volume_multiple 必须大于 0。")
    if not 0 <= cfg.min_body_ratio <= 1:
        raise ValueError("min_body_ratio 必须在 0 到 1 之间。")

    result = bars.sort_values(["stock_code", "date"]).reset_index(drop=True).copy()
    grouped = result.groupby("stock_code", sort=False)
    candle_range = (result["high"] - result["low"]).astype(float)
    candle_body = (result["close"] - result["open"]).abs().astype(float)
    result["_range_mean"] = candle_range.groupby(result["stock_code"]).transform(
        lambda series: series.rolling(cfg.lookback, min_periods=2).mean().shift(1)
    )
    result["_volume_mean"] = grouped["volume"].transform(
        lambda series: pd.to_numeric(series, errors="coerce").rolling(cfg.lookback, min_periods=2).mean().shift(1)
    )
    body_ratio = candle_body / candle_range.replace(0, pd.NA)
    range_pass = candle_range >= result["_range_mean"] * cfg.range_multiple
    volume_pass = pd.to_numeric(result["volume"], errors="coerce") >= result["_volume_mean"] * cfg.volume_multiple
    body_pass = body_ratio.fillna(0.0) >= cfg.min_body_ratio

    result["is_landmark"] = (range_pass & volume_pass & body_pass).fillna(False)
    result["landmark_range_ratio"] = (candle_range / result["_range_mean"]).replace([float("inf"), -float("inf")], pd.NA)
    result["landmark_volume_ratio"] = (
        pd.to_numeric(result["volume"], errors="coerce") / result["_volume_mean"]
    ).replace([float("inf"), -float("inf")], pd.NA)
    result["landmark_body_ratio"] = body_ratio.fillna(0.0)
    result["landmark_reason"] = ""
    result.loc[result["is_landmark"], "landmark_reason"] = "range+volume+body"
    return result.drop(columns=["_range_mean", "_volume_mean"])
