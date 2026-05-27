from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from trending_winning.data.repository import load_multi_timeframe_backtest_data
from trending_winning.strategy import StrategyConfig, scan_bars


@dataclass(frozen=True)
class MultiTimeframeScanResult:
    """多周期扫描结果；full 保存全历史，latest 保存每票每周期最新一根。"""

    full: pd.DataFrame
    latest: pd.DataFrame
    data_audit: pd.DataFrame
    filtered_limit_open_days: pd.DataFrame


def scan_timeframes(
    *,
    data_root: str | Path,
    timeframes: tuple[str, ...] | list[str],
    adjust: str,
    symbols: tuple[str, ...] | list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    strategy: StrategyConfig | None = None,
    strict_data_quality: bool = True,
    min_coverage_ratio: float | None = None,
) -> MultiTimeframeScanResult:
    bundle = load_multi_timeframe_backtest_data(
        data_root=data_root,
        timeframes=tuple(timeframes),
        adjust=adjust,
        symbols=tuple(symbols),
        start=start,
        end=end,
        strict_data_quality=strict_data_quality,
        min_coverage_ratio=min_coverage_ratio,
    )
    frames: list[pd.DataFrame] = []
    for timeframe, bars in bundle.bars_by_timeframe.items():
        scanned = scan_bars(bars, strategy or StrategyConfig())
        if scanned.empty:
            continue
        scanned = scanned.copy()
        scanned.insert(0, "timeframe", timeframe)
        frames.append(scanned)

    if not frames:
        empty = pd.DataFrame()
        return MultiTimeframeScanResult(
            full=empty,
            latest=empty,
            data_audit=bundle.data_audit,
            filtered_limit_open_days=bundle.filtered_limit_open_days,
        )

    full = pd.concat(frames, ignore_index=True)
    latest = (
        full.sort_values(["timeframe", "stock_code", "date"])
        .groupby(["timeframe", "stock_code"], sort=True)
        .tail(1)
        .reset_index(drop=True)
    )
    return MultiTimeframeScanResult(
        full=full,
        latest=latest,
        data_audit=bundle.data_audit,
        filtered_limit_open_days=bundle.filtered_limit_open_days,
    )
