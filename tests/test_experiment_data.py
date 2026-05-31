from __future__ import annotations

import sys

import pandas as pd

from trending_winning.backtest.experiment_models import SingleStrategyExperimentConfig
from trending_winning.backtest.models import BacktestResult


class FakeExperimentRepository:
    def __init__(self) -> None:
        self.inventory_calls: list[dict[str, object]] = []
        self.single_calls: list[dict[str, object]] = []
        self.multi_calls: list[dict[str, object]] = []

    def inventory(self, **kwargs: object) -> pd.DataFrame:
        self.inventory_calls.append(kwargs)
        return pd.DataFrame(
            {
                "stock_code": ["000001.SZ", "000001.SZ"],
                "timeframe": ["1d", "30m"],
                "status": ["cached", "cached"],
            }
        )

    def load_backtest_data(self, **kwargs: object) -> object:
        self.single_calls.append(kwargs)
        return _bundle(bars=_bars("30m"), higher_bars=pd.DataFrame())

    def load_multi_timeframe_backtest_data(self, **kwargs: object) -> object:
        self.multi_calls.append(kwargs)
        return _multi_bundle()


class FakeBundle:
    def __init__(self, bars: pd.DataFrame, higher_bars: pd.DataFrame) -> None:
        self.bars = bars
        self.bars_by_timeframe = {"30m": bars, "60m": higher_bars}
        self.data_audit = pd.DataFrame({"stock_code": ["000001.SZ"], "status": ["ok"]})
        self.data_inventory = pd.DataFrame(
            {"stock_code": ["000001.SZ"], "timeframe": ["30m"], "status": ["cached"]}
        )
        self.limit_filter_audit = pd.DataFrame({"stock_code": ["000001.SZ"], "status": ["ok"], "filtered_days": [1]})
        self.filtered_limit_open_days = pd.DataFrame({"stock_code": ["000001.SZ"], "date": [pd.Timestamp("2026-05-25")]})


def test_load_experiment_data_imports_without_experiment_runner() -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_data import experiment_inventory_timeframes, load_experiment_data

    repo = FakeExperimentRepository()
    config = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        higher_timeframe="60m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        strict_data_quality=False,
        min_coverage_ratio=0.5,
    )

    data = load_experiment_data(repo, config)

    assert "trending_winning.backtest.experiment" not in sys.modules
    assert experiment_inventory_timeframes(config) == ["1d", "30m", "60m"]
    assert repo.inventory_calls == [{"timeframes": ("1d", "30m", "60m"), "symbols": ("000001.SZ",)}]
    assert repo.single_calls == []
    assert repo.multi_calls == [
        {
            "timeframes": ("30m", "60m"),
            "symbols": ("000001.SZ",),
            "start": "2026-05-25",
            "end": "2026-05-25",
            "strict_data_quality": False,
            "min_coverage_ratio": 0.5,
        }
    ]
    assert data.bars["timeframe"].tolist() == ["30m"]
    assert data.higher_bars["timeframe"].tolist() == ["60m"]
    assert data.data_inventory["status"].tolist() == ["cached", "cached"]


def test_with_data_management_statistics_uses_loaded_data_summary() -> None:
    from trending_winning.backtest.experiment_data import with_data_management_statistics

    result = BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"trade_count": 0})
    data = _bundle(bars=_bars("30m"), higher_bars=pd.DataFrame())

    updated = with_data_management_statistics(result, data, min_coverage_ratio=0.8)

    assert updated.stats["trade_count"] == 0
    assert updated.stats["data_audit_row_count"] == 1.0
    assert updated.stats["limit_filter_audit_row_count"] == 1.0
    assert updated.stats["limit_filter_filtered_days"] == 1.0


def _bundle(*, bars: pd.DataFrame, higher_bars: pd.DataFrame) -> FakeBundle:
    return FakeBundle(bars=bars, higher_bars=higher_bars)


def _multi_bundle() -> FakeBundle:
    return FakeBundle(bars=_bars("30m"), higher_bars=_bars("60m"))


def _bars(timeframe: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-05-25 09:30:00")],
            "stock_code": ["000001.SZ"],
            "timeframe": [timeframe],
            "open": [10.0],
            "high": [10.2],
            "low": [9.8],
            "close": [10.1],
            "volume": [1000.0],
            "amount": [10100.0],
        }
    )
