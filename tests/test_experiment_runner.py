from __future__ import annotations

from inspect import getsource
import json
from pathlib import Path

import pandas as pd
import pytest

from trending_winning.backtest.experiment import (
    PortfolioExperimentConfig,
    SingleStrategyExperimentConfig,
    benchmark_portfolio_experiment,
    run_portfolio_parameter_sweep,
    run_portfolio_experiment,
    run_single_strategy_parameter_sweep,
    run_single_strategy_experiment,
)
from trending_winning.backtest.engine import BacktestResult
from trending_winning.backtest import experiment as experiment_module
from trending_winning.backtest import portfolio as portfolio_module
from trending_winning.data.repository import write_local_bars
from trending_winning.strategies.base import ORDER_COLUMNS


def test_experiment_module_does_not_import_trend_detector_at_module_load() -> None:
    source_before_config = getsource(experiment_module).split("DATA_SCOPE_SWEEP_FIELDS", maxsplit=1)[0]

    assert "trending_winning.detectors.trend" not in source_before_config


def test_experiment_json_ready_replaces_non_finite_numbers_for_strict_json() -> None:
    payload = experiment_module._json_ready(
        {
            "profit_factor": float("inf"),
            "loss_factor": float("-inf"),
            "nan_metric": float("nan"),
            "nested": [1.0, float("inf")],
        }
    )

    assert payload == {
        "profit_factor": None,
        "loss_factor": None,
        "nan_metric": None,
        "nested": [1.0, None],
    }
    json.dumps(payload, allow_nan=False)


def test_portfolio_experiment_saves_reproducible_config_and_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    output_dir = tmp_path / "runs" / "case-001"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00", "2026-05-25 11:00:00"]
            ),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2, 10.4, 10.6],
            "high": [10.3, 10.5, 10.7, 10.9],
            "low": [9.8, 10.0, 10.2, 10.4],
            "close": [10.2, 10.4, 10.6, 10.8],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "amount": [10200.0, 11440.0, 12720.0, 14040.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    config = PortfolioExperimentConfig(
        name="case-001",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend", "range"),
        max_holding_bars=3,
        max_open_positions=2,
        short_margin_rate=1.5,
        intrabar_exit_policy="optimistic",
        reversal_old_extreme_tolerance_pct=0.02,
        reversal_require_old_extreme_test=False,
        reversal_require_structure_confirmation=False,
        strict_data_quality=False,
        output_dir=str(output_dir),
    )

    result = run_portfolio_experiment(config, save=True)

    assert result.config.name == "case-001"
    assert result.backtest.equity_curve is not None
    assert (output_dir / "config.json").exists()
    assert (output_dir / "stats.json").exists()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "order_decisions.csv").exists()
    assert (output_dir / "strategy_filter_decisions.csv").exists()
    assert (output_dir / "order_decision_stats.csv").exists()
    assert (output_dir / "strategy_filter_stats.csv").exists()
    assert (output_dir / "equity_curve.csv").exists()
    assert (output_dir / "data_coverage.csv").exists()
    assert (output_dir / "limit_filter_audit.csv").exists()
    assert (output_dir / "strategy_stats.csv").exists()
    assert (output_dir / "symbol_stats.csv").exists()
    assert (output_dir / "side_stats.csv").exists()
    assert (output_dir / "exit_reason_stats.csv").exists()
    assert (output_dir / "event_type_stats.csv").exists()
    assert (output_dir / "monthly_returns.csv").exists()
    saved_config = json.loads((output_dir / "config.json").read_text())
    saved_stats = json.loads((output_dir / "stats.json").read_text())
    saved_coverage = pd.read_csv(output_dir / "data_coverage.csv")
    saved_limit_filter_audit = pd.read_csv(output_dir / "limit_filter_audit.csv")
    saved_decisions = pd.read_csv(output_dir / "order_decisions.csv")
    saved_filter_decisions = pd.read_csv(output_dir / "strategy_filter_decisions.csv")
    saved_decision_stats = pd.read_csv(output_dir / "order_decision_stats.csv")
    saved_filter_stats = pd.read_csv(output_dir / "strategy_filter_stats.csv")
    assert saved_config["detectors"] == ["trend", "range"]
    assert saved_config["short_margin_rate"] == 1.5
    assert saved_config["intrabar_exit_policy"] == "optimistic"
    assert saved_config["reversal_old_extreme_tolerance_pct"] == 0.02
    assert saved_config["reversal_require_old_extreme_test"] is False
    assert saved_config["reversal_require_structure_confirmation"] is False
    assert saved_stats["trade_count"] == result.backtest.stats["trade_count"]
    assert saved_stats["strategy_signal_count"] == result.backtest.stats["strategy_signal_count"]
    assert result.backtest.stats["data_audit_row_count"] == 1.0
    assert result.backtest.stats["limit_filter_audit_row_count"] == 1.0
    assert result.backtest.stats["limit_filter_failed_count"] == 1.0
    assert saved_stats["data_audit_row_count"] == result.backtest.stats["data_audit_row_count"]
    assert saved_stats["limit_filter_failed_count"] == result.backtest.stats["limit_filter_failed_count"]
    assert result.data_coverage["status"].tolist() == ["ok"]
    assert saved_coverage.loc[0, "stock_code"] == "000001.SZ"
    assert result.limit_filter_audit["status"].tolist() == ["daily_missing"]
    assert saved_limit_filter_audit.loc[0, "status"] == "daily_missing"
    assert {"order_id", "status", "reason"}.issubset(saved_decisions.columns)
    assert {"order_id", "status", "reason"}.issubset(saved_filter_decisions.columns)
    assert {"strategy_name", "detector_name", "status", "reason", "decision_count", "decision_rate"}.issubset(
        saved_decision_stats.columns
    )
    assert {"strategy_name", "filter_name", "status", "reason", "decision_count", "decision_rate"}.issubset(
        saved_filter_stats.columns
    )
    assert {"status", "reason", "decision_count"}.issubset(result.order_decision_stats.columns)
    assert {"status", "reason", "decision_count"}.issubset(result.strategy_filter_stats.columns)
    assert "strategy_name" in result.strategy_stats.columns
    assert "stock_code" in result.symbol_stats.columns
    assert "side" in result.side_stats.columns
    assert "exit_reason" in result.exit_reason_stats.columns
    assert "event_type" in result.event_type_stats.columns
    assert result.monthly_returns["period"].tolist() == ["2026-05"]


def test_benchmark_portfolio_experiment_reports_throughput_and_saves_json(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    output_dir = tmp_path / "runs" / "bench"
    rows: list[dict[str, object]] = []
    for symbol in ("000001.SZ", "000002.SZ"):
        for index in range(20):
            close = 10.0 + index * 0.05
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": close - 0.05,
                    "high": close + 0.15,
                    "low": close - 0.15,
                    "close": close,
                    "volume": 1000.0 + index,
                    "amount": close * (1000.0 + index),
                }
            )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))
    config = PortfolioExperimentConfig(
        name="bench",
        data_root=str(data_root),
        symbols=("000001.SZ", "000002.SZ"),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend", "range"),
        strict_data_quality=False,
        output_dir=str(output_dir),
    )

    report = benchmark_portfolio_experiment(config, save=True)

    assert report.bar_count == 40
    assert report.elapsed_seconds > 0
    assert report.bars_per_second > 0
    assert report.trade_count >= 0
    assert (output_dir / "benchmark.json").exists()
    saved_report = json.loads((output_dir / "benchmark.json").read_text())
    assert saved_report["bar_count"] == 40


def test_portfolio_experiment_can_explicitly_disable_strict_data_quality(tmp_path: Path) -> None:
    data_root = tmp_path / "market" / "daily"
    root = data_root.parent / "30m" / "qfq"
    root.mkdir(parents=True)
    bad = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 10:00:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000003.SZ", "000003.SZ"],
            "open": [0.0, 10.0],
            "high": [10.2, 10.2],
            "low": [9.8, 9.8],
            "close": [10.1, pd.NA],
            "volume": [1000.0, 1000.0],
            "amount": [10100.0, 10100.0],
        }
    )
    bad.to_parquet(root / "000003.SZ.parquet", index=False)
    config = PortfolioExperimentConfig(
        name="quality-off",
        data_root=str(data_root),
        symbols=("000003.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        strict_data_quality=False,
    )

    result = run_portfolio_experiment(config)

    assert result.data_coverage.loc[0, "status"] == "quality_error"
    assert result.input_bar_count == 1


def test_portfolio_experiment_can_gate_orders_by_higher_timeframe_trend(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    base_rows: list[dict[str, object]] = []
    for symbol, prices in {
        "000001.SZ": [10.0, 10.4, 10.7],
        "000002.SZ": [20.0, 19.6, 19.4],
    }.items():
        for index, close in enumerate(prices):
            base_rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 10:45:00") + pd.Timedelta(minutes=15 * index),
                    "stock_code": symbol,
                    "open": close - 0.1,
                    "high": close + 0.3,
                    "low": close - 0.3,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    higher_rows: list[dict[str, object]] = []
    for symbol, prices in {
        "000001.SZ": [10.0, 10.3, 10.6, 10.9],
        "000002.SZ": [20.0, 19.7, 19.4, 19.1],
    }.items():
        for index, close in enumerate(prices):
            higher_rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:00:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": close - 0.05,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    write_local_bars(data_root=data_root, timeframe="15m", adjust="qfq", bars=pd.DataFrame(base_rows))
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=pd.DataFrame(higher_rows))

    class TwoLongOrdersStrategy:
        name = "fixed_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "order_id": "aligned-long",
                        "strategy_name": self.name,
                        "detector_name": "fixed",
                        "event_id": "event:aligned-long",
                        "stock_code": "000001.SZ",
                        "timeframe": timeframe,
                        "signal_date": pd.Timestamp("2026-05-25 10:45:00"),
                        "signal_bar_index": 0,
                        "side": "long",
                        "signal_price": 10.0,
                        "entry_price": 10.2,
                        "stop_price": 9.8,
                        "target_price": 11.0,
                        "max_holding_bars": 2,
                        "max_actual_risk_pct": None,
                        "max_chase_pct": None,
                        "metadata": {},
                    },
                    {
                        "order_id": "blocked-long",
                        "strategy_name": self.name,
                        "detector_name": "fixed",
                        "event_id": "event:blocked-long",
                        "stock_code": "000002.SZ",
                        "timeframe": timeframe,
                        "signal_date": pd.Timestamp("2026-05-25 10:45:00"),
                        "signal_bar_index": 0,
                        "side": "long",
                        "signal_price": 20.0,
                        "entry_price": 20.2,
                        "stop_price": 19.8,
                        "target_price": 21.0,
                        "max_holding_bars": 2,
                        "max_actual_risk_pct": None,
                        "max_chase_pct": None,
                        "metadata": {},
                    },
                ],
                columns=ORDER_COLUMNS,
            )

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", lambda cfg: [TwoLongOrdersStrategy()])
    config = PortfolioExperimentConfig(
        name="mtf-gated",
        data_root=str(data_root),
        symbols=("000001.SZ", "000002.SZ"),
        timeframe="15m",
        higher_timeframe="60m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend",),
        trend_lookback=3,
        trend_min_score=0.01,
        min_coverage_ratio=0.1,
        strict_data_quality=False,
    )

    result = run_portfolio_experiment(config)

    assert result.backtest.trades["order_id"].tolist() == ["aligned-long"]
    assert result.backtest.trades.loc[0, "strategy_name"] == "fixed_signal_bar_mtf_60m"
    assert result.backtest.trades.loc[0, "metadata"]["higher_timeframe"] == "60m"
    filter_decisions = result.backtest.strategy_filter_decisions.set_index("order_id")
    assert filter_decisions.loc["aligned-long", "status"] == "accepted"
    assert filter_decisions.loc["blocked-long", "reason"] == "higher_timeframe_mismatch"
    assert result.backtest.stats["strategy_signal_count"] == 2.0
    assert result.backtest.stats["strategy_rejected_higher_timeframe_mismatch_count"] == 1.0
    assert set(result.data_coverage["timeframe"]) == {"15m", "60m"}


def test_single_strategy_experiment_passes_cost_model_to_backtest_config(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    captured: dict[str, float] = {}

    def spy_single_backtest(*args, **kwargs):
        cfg = args[2]
        captured["fee_rate"] = cfg.fee_rate
        captured["slippage_bps"] = cfg.slippage_bps
        captured["initial_equity"] = cfg.initial_equity
        trades = pd.DataFrame(columns=["strategy_name", "stock_code", "return_pct", "holding_bars"])
        return BacktestResult(trades=trades, equity_curve=pd.DataFrame(), stats={"trade_count": 0.0})

    monkeypatch.setattr(experiment_module, "run_single_strategy_backtest", spy_single_backtest)
    config = SingleStrategyExperimentConfig(
        name="single-cost",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        fee_rate=0.0003,
        slippage_bps=5.0,
        initial_equity=2.0,
        strict_data_quality=False,
    )

    run_single_strategy_experiment(config)

    assert captured == {"fee_rate": 0.0003, "slippage_bps": 5.0, "initial_equity": 2.0}


def test_portfolio_experiment_passes_cost_model_to_backtest_config(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    captured: dict[str, float] = {}

    def spy_portfolio_backtest(*args, **kwargs):
        cfg = args[2]
        captured["fee_rate"] = cfg.fee_rate
        captured["slippage_bps"] = cfg.slippage_bps
        captured["initial_equity"] = cfg.initial_equity
        trades = pd.DataFrame(columns=["strategy_name", "stock_code", "return_pct", "holding_bars"])
        return BacktestResult(trades=trades, equity_curve=pd.DataFrame(), stats={"trade_count": 0.0})

    monkeypatch.setattr(experiment_module, "run_portfolio_backtest", spy_portfolio_backtest)
    config = PortfolioExperimentConfig(
        name="portfolio-cost",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        fee_rate=0.0003,
        slippage_bps=5.0,
        initial_equity=2.0,
        strict_data_quality=False,
    )

    run_portfolio_experiment(config)

    assert captured == {"fee_rate": 0.0003, "slippage_bps": 5.0, "initial_equity": 2.0}


def test_portfolio_experiment_passes_allocation_limits_to_portfolio_config(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    captured: dict[str, object] = {}

    def spy_portfolio_backtest(*args, **kwargs):
        pcfg = args[3]
        captured["capital_per_trade"] = pcfg.capital_per_trade
        captured["reserve_cash"] = pcfg.reserve_cash
        captured["allow_same_symbol_overlap"] = pcfg.allow_same_symbol_overlap
        captured["strategy_priority"] = dict(pcfg.strategy_priority)
        captured["strategy_capital_limit"] = dict(pcfg.strategy_capital_limit)
        captured["sector_capital_limit"] = dict(pcfg.sector_capital_limit)
        captured["symbol_sector_map"] = dict(pcfg.symbol_sector_map)
        captured["sector_metadata_key"] = pcfg.sector_metadata_key
        captured["default_sector"] = pcfg.default_sector
        trades = pd.DataFrame(columns=["strategy_name", "stock_code", "return_pct", "holding_bars"])
        return BacktestResult(trades=trades, equity_curve=pd.DataFrame(), stats={"trade_count": 0.0})

    monkeypatch.setattr(experiment_module, "run_portfolio_backtest", spy_portfolio_backtest)
    config = PortfolioExperimentConfig(
        name="allocation-limits",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        capital_per_trade=0.25,
        reserve_cash=0.1,
        allow_same_symbol_overlap=True,
        strategy_priority={"trend_signal_bar": 1},
        strategy_capital_limit={"trend_signal_bar": 0.6},
        sector_capital_limit={"银行": 0.5},
        symbol_sector_map={"000001.SZ": "银行"},
        sector_metadata_key="industry",
        default_sector="未分类",
        strict_data_quality=False,
    )

    run_portfolio_experiment(config)

    assert captured == {
        "capital_per_trade": 0.25,
        "reserve_cash": 0.1,
        "allow_same_symbol_overlap": True,
        "strategy_priority": {"trend_signal_bar": 1},
        "strategy_capital_limit": {"trend_signal_bar": 0.6},
        "sector_capital_limit": {"银行": 0.5},
        "symbol_sector_map": {"000001.SZ": "银行"},
        "sector_metadata_key": "industry",
        "default_sector": "未分类",
    }


def test_portfolio_experiment_passes_detector_parameters_to_strategy_suite(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    captured: dict[str, object] = {}

    def spy_suite(cfg):
        captured["trend_lookback"] = cfg.trend_lookback
        captured["trend_min_score"] = cfg.trend_min_score
        captured["trend_strong_close_pos"] = cfg.trend_strong_close_pos
        captured["trend_min_body_ratio"] = cfg.trend_min_body_ratio
        captured["trend_pullback_lookback"] = cfg.trend_pullback_lookback
        captured["trend_h2_min_pullback_legs"] = cfg.trend_h2_min_pullback_legs
        captured["range_lookback"] = cfg.range_lookback
        captured["range_middle_low"] = cfg.range_middle_low
        captured["range_middle_high"] = cfg.range_middle_high
        captured["range_false_break_buffer"] = cfg.range_false_break_buffer
        captured["range_strong_close_pos"] = cfg.range_strong_close_pos
        captured["range_min_score"] = cfg.range_min_score
        captured["channel_lookback"] = cfg.channel_lookback
        captured["channel_sigma_multiple"] = cfg.channel_sigma_multiple
        captured["channel_method"] = cfg.channel_method
        captured["channel_break_buffer"] = cfg.channel_break_buffer
        captured["channel_swing_left_bars"] = cfg.channel_swing_left_bars
        captured["channel_swing_right_bars"] = cfg.channel_swing_right_bars
        captured["reversal_lookback"] = cfg.reversal_lookback
        captured["reversal_strong_close_pos"] = cfg.reversal_strong_close_pos
        captured["reversal_min_body_ratio"] = cfg.reversal_min_body_ratio
        captured["reversal_old_extreme_tolerance_pct"] = cfg.reversal_old_extreme_tolerance_pct
        return []

    def spy_portfolio_backtest(*args, **kwargs):
        trades = pd.DataFrame(columns=["strategy_name", "stock_code", "return_pct", "holding_bars"])
        return BacktestResult(trades=trades, equity_curve=pd.DataFrame(), stats={"trade_count": 0.0})

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", spy_suite)
    monkeypatch.setattr(experiment_module, "run_portfolio_backtest", spy_portfolio_backtest)
    config = PortfolioExperimentConfig(
        name="detector-params",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        trend_lookback=7,
        trend_min_score=0.4,
        trend_strong_close_pos=0.7,
        trend_min_body_ratio=0.55,
        trend_pullback_lookback=4,
        trend_h2_min_pullback_legs=3,
        range_lookback=8,
        range_middle_low=0.2,
        range_middle_high=0.8,
        range_false_break_buffer=0.01,
        range_strong_close_pos=0.7,
        range_min_score=0.9,
        channel_lookback=9,
        channel_sigma_multiple=1.6,
        channel_method="swing",
        channel_break_buffer=0.02,
        channel_swing_left_bars=3,
        channel_swing_right_bars=4,
        reversal_lookback=10,
        reversal_strong_close_pos=0.7,
        reversal_min_body_ratio=0.5,
        reversal_old_extreme_tolerance_pct=0.03,
        strict_data_quality=False,
    )

    run_portfolio_experiment(config)

    assert captured == {
        "trend_lookback": 7,
        "trend_min_score": 0.4,
        "trend_strong_close_pos": 0.7,
        "trend_min_body_ratio": 0.55,
        "trend_pullback_lookback": 4,
        "trend_h2_min_pullback_legs": 3,
        "range_lookback": 8,
        "range_middle_low": 0.2,
        "range_middle_high": 0.8,
        "range_false_break_buffer": 0.01,
        "range_strong_close_pos": 0.7,
        "range_min_score": 0.9,
        "channel_lookback": 9,
        "channel_sigma_multiple": 1.6,
        "channel_method": "swing",
        "channel_break_buffer": 0.02,
        "channel_swing_left_bars": 3,
        "channel_swing_right_bars": 4,
        "reversal_lookback": 10,
        "reversal_strong_close_pos": 0.7,
        "reversal_min_body_ratio": 0.5,
        "reversal_old_extreme_tolerance_pct": 0.03,
    }


def test_portfolio_parameter_sweep_reuses_loaded_data_and_saves_ranked_table(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    output_dir = tmp_path / "runs" / "sweep"
    rows: list[dict[str, object]] = []
    for symbol in ("000001.SZ", "000002.SZ"):
        for index in range(12):
            close = 10.0 + index * 0.08
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.15,
                    "close": close,
                    "volume": 1000.0 + index,
                    "amount": close * (1000.0 + index),
                }
            )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))
    original_load = experiment_module.MarketDataRepository.load_backtest_data
    load_calls = 0

    def spy_load(self, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(self, **kwargs)

    monkeypatch.setattr(experiment_module.MarketDataRepository, "load_backtest_data", spy_load)
    config = PortfolioExperimentConfig(
        name="sweep",
        data_root=str(data_root),
        symbols=("000001.SZ", "000002.SZ"),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend", "range"),
        strict_data_quality=False,
        output_dir=str(output_dir),
    )

    result = run_portfolio_parameter_sweep(
        config,
        grid={"risk_reward": [1.5, 2.0], "max_holding_bars": [3, 5]},
        save=True,
    )

    assert load_calls == 1
    assert len(result.table) == 4
    assert {
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_config_hash",
        "case_name",
        "risk_reward",
        "max_holding_bars",
        "trade_count",
        "total_return",
        "bars_per_second",
        "order_cache_status",
        "candidate_cache_status",
        "generated_order_count",
        "candidate_count",
        "candidate_rejection_count",
        "data_weighted_coverage_ratio",
        "filtered_limit_open_count",
    }.issubset(result.table.columns)
    assert result.table["sweep_rank"].tolist() == [1, 2, 3, 4]
    assert result.table["case_config_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.table["case_config_hash"].is_unique
    assert result.table.columns[0] == "sweep_rank"
    assert result.table.columns[1] == "pareto_rank"
    assert result.table.columns[2] == "is_pareto_efficient"
    assert result.table.columns[3] == "case_config_hash"
    assert result.table["total_return"].tolist() == sorted(result.table["total_return"].tolist(), reverse=True)
    assert result.data_coverage["status"].tolist() == ["ok", "ok"]
    assert (output_dir / "sweep.csv").exists()
    assert (output_dir / "config.json").exists()
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config["name"] == "sweep"
    assert saved_config["sweep_grid"] == {"risk_reward": [1.5, 2.0], "max_holding_bars": [3, 5]}
    saved_sweep = pd.read_csv(output_dir / "sweep.csv")
    assert {
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_config_hash",
        "order_count",
        "accepted_order_count",
        "rejected_order_count",
        "acceptance_rate",
        "rejected_no_fill_count",
        "order_cache_status",
        "candidate_cache_status",
        "generated_order_count",
        "candidate_count",
        "candidate_rejection_count",
        "data_weighted_coverage_ratio",
        "filtered_limit_open_count",
    }.issubset(saved_sweep.columns)
    assert saved_sweep["sweep_rank"].tolist() == [1, 2, 3, 4]
    assert saved_sweep["case_config_hash"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_portfolio_parameter_sweep_rejects_data_scope_grid_fields(tmp_path: Path, monkeypatch) -> None:
    def fail_load(self, **kwargs):
        raise AssertionError("数据范围字段非法时不应加载行情")

    monkeypatch.setattr(experiment_module.MarketDataRepository, "load_backtest_data", fail_load)
    config = PortfolioExperimentConfig(
        name="sweep-data-scope",
        data_root=str(tmp_path / "market" / "daily"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
    )

    with pytest.raises(ValueError, match="不能在同一次 sweep 中改变数据范围字段.*timeframe"):
        run_portfolio_parameter_sweep(config, grid={"timeframe": ["30m", "60m"]})


def test_portfolio_parameter_sweep_reuses_orders_when_only_portfolio_params_change(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    generate_calls = 0

    class CountingStrategy:
        name = "counting_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            nonlocal generate_calls
            generate_calls += 1
            return pd.DataFrame(columns=ORDER_COLUMNS)

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", lambda cfg: [CountingStrategy()])
    config = PortfolioExperimentConfig(
        name="sweep-order-cache",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend",),
        strict_data_quality=False,
    )

    result = run_portfolio_parameter_sweep(
        config,
        grid={"max_open_positions": [1, 2, 3], "reserve_cash": [0.0, 0.1]},
    )

    assert len(result.table) == 6
    assert generate_calls == 1
    by_case = result.table.sort_values("case_name").set_index("case_name")
    assert by_case["order_cache_status"].tolist() == ["miss", "hit", "hit", "hit", "hit", "hit"]
    assert by_case["generated_order_count"].tolist() == [0, 0, 0, 0, 0, 0]


def test_portfolio_parameter_sweep_reuses_candidate_trades_when_only_portfolio_params_change(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 10.4, 10.8],
            "high": [10.2, 10.7, 11.0],
            "low": [9.8, 10.2, 10.6],
            "close": [10.0, 10.6, 10.9],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10000.0, 11660.0, 13080.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)

    class OneOrderStrategy:
        name = "trend_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "order_id": "order-1",
                        "strategy_name": self.name,
                        "detector_name": "trend",
                        "event_id": "event-1",
                        "stock_code": "000001.SZ",
                        "timeframe": timeframe,
                        "signal_date": pd.Timestamp("2026-05-25 09:30:00"),
                        "signal_bar_index": 0,
                        "side": "long",
                        "entry_price": 10.4,
                        "stop_price": 10.0,
                        "target_price": 11.2,
                        "max_holding_bars": 2,
                        "metadata": {},
                    }
                ],
                columns=ORDER_COLUMNS,
            )

    simulate_calls = 0

    def spy_simulate_order_trade(group, order, signal_index, cfg):
        nonlocal simulate_calls
        simulate_calls += 1
        return {
            "order_id": order["order_id"],
            "event_id": order["event_id"],
            "strategy_name": order["strategy_name"],
            "detector_name": order["detector_name"],
            "stock_code": order["stock_code"],
            "timeframe": order["timeframe"],
            "signal_date": order["signal_date"],
            "signal_bar_index": int(signal_index),
            "side": order["side"],
            "planned_entry_price": float(order["entry_price"]),
            "entry_date": pd.Timestamp("2026-05-25 10:00:00"),
            "entry_price": 10.4,
            "stop_price": 10.0,
            "target_price": 11.2,
            "risk_per_share": 0.4,
            "exit_date": pd.Timestamp("2026-05-25 10:30:00"),
            "exit_price": 10.8,
            "exit_reason": "max_holding",
            "holding_bars": 1,
            "return_pct": 3.846153846154,
            "r_multiple": 1.0,
            "mae_pct": 0.0,
            "mfe_pct": 3.846153846154,
            "mae_r": 0.0,
            "mfe_r": 1.0,
            "metadata": {},
            "_exit_index": 2,
        }, ""

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", lambda cfg: [OneOrderStrategy()])
    monkeypatch.setattr(portfolio_module, "simulate_order_trade_with_rejection", spy_simulate_order_trade)
    config = PortfolioExperimentConfig(
        name="sweep-candidate-cache",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend",),
        strict_data_quality=False,
    )

    result = run_portfolio_parameter_sweep(
        config,
        grid={"max_open_positions": [1, 2, 3], "reserve_cash": [0.0, 0.1]},
    )

    assert len(result.table) == 6
    assert simulate_calls == 1
    by_case = result.table.sort_values("case_name").set_index("case_name")
    assert by_case["candidate_cache_status"].tolist() == ["miss", "hit", "hit", "hit", "hit", "hit"]
    assert by_case["candidate_count"].tolist() == [1, 1, 1, 1, 1, 1]
    assert by_case["candidate_rejection_count"].tolist() == [0, 0, 0, 0, 0, 0]


def test_portfolio_parameter_sweep_does_not_reuse_orders_when_higher_timeframe_gate_changes(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    base_bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-05-25 10:00:00",
                    "2026-05-25 10:15:00",
                    "2026-05-25 11:00:00",
                    "2026-05-25 11:15:00",
                ]
            ),
            "stock_code": ["000001.SZ"] * 4,
            "open": [10.0, 10.2, 10.8, 11.0],
            "high": [10.1, 10.6, 10.9, 11.4],
            "low": [9.9, 10.1, 10.7, 10.9],
            "close": [10.0, 10.5, 10.8, 11.2],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            "amount": [10000.0, 11550.0, 12960.0, 14560.0],
        }
    )
    higher_bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00"]),
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "high": [10.3],
            "low": [9.9],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="15m", adjust="qfq", bars=base_bars)
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=higher_bars)

    class TwoAgedOrdersStrategy:
        name = "trend_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "order_id": "fresh-context",
                        "strategy_name": self.name,
                        "detector_name": "trend",
                        "event_id": "event:fresh-context",
                        "stock_code": "000001.SZ",
                        "timeframe": timeframe,
                        "signal_date": pd.Timestamp("2026-05-25 10:00:00"),
                        "signal_bar_index": 0,
                        "side": "long",
                        "signal_price": 10.0,
                        "entry_price": 10.2,
                        "stop_price": 9.8,
                        "target_price": 11.5,
                        "max_holding_bars": 1,
                        "max_actual_risk_pct": None,
                        "max_chase_pct": None,
                        "metadata": {},
                    },
                    {
                        "order_id": "older-context",
                        "strategy_name": self.name,
                        "detector_name": "trend",
                        "event_id": "event:older-context",
                        "stock_code": "000001.SZ",
                        "timeframe": timeframe,
                        "signal_date": pd.Timestamp("2026-05-25 11:00:00"),
                        "signal_bar_index": 2,
                        "side": "long",
                        "signal_price": 10.8,
                        "entry_price": 11.0,
                        "stop_price": 10.6,
                        "target_price": 12.0,
                        "max_holding_bars": 1,
                        "max_actual_risk_pct": None,
                        "max_chase_pct": None,
                        "metadata": {},
                    },
                ],
                columns=ORDER_COLUMNS,
            )

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", lambda cfg: [TwoAgedOrdersStrategy()])
    monkeypatch.setattr(
        experiment_module,
        "_higher_timeframe_context",
        lambda bars, config: pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-05-25 09:30:00")],
                "stock_code": ["000001.SZ"],
                "trend_state": ["bull"],
            }
        ),
    )
    config = PortfolioExperimentConfig(
        name="sweep-mtf-age",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="15m",
        higher_timeframe="60m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend",),
        strict_data_quality=False,
    )

    result = run_portfolio_parameter_sweep(
        config,
        grid={"higher_timeframe_max_age_minutes": [30, 120]},
    )

    by_age = result.table.sort_values("higher_timeframe_max_age_minutes").set_index(
        "higher_timeframe_max_age_minutes"
    )
    assert by_age.loc[30, "generated_order_count"] == 1
    assert by_age.loc[120, "generated_order_count"] == 2
    assert by_age.loc[30, "trade_count"] == 1
    assert by_age.loc[120, "trade_count"] == 2
    assert by_age["order_cache_status"].tolist() == ["miss", "miss"]


def test_portfolio_parameter_sweep_uses_loaded_normalized_bars_without_portfolio_renormalization(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    normalize_calls = 0
    original_normalize = portfolio_module.normalize_bars

    def spy_normalize_bars(frame: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
        nonlocal normalize_calls
        normalize_calls += 1
        return original_normalize(frame, symbol)

    class EmptyStrategy:
        name = "empty_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            return pd.DataFrame(columns=ORDER_COLUMNS)

    monkeypatch.setattr(portfolio_module, "normalize_bars", spy_normalize_bars)
    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", lambda cfg: [EmptyStrategy()])
    config = PortfolioExperimentConfig(
        name="sweep-normalized-bars",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detectors=("trend",),
        strict_data_quality=False,
    )

    result = run_portfolio_parameter_sweep(
        config,
        grid={"max_open_positions": [1, 2, 3], "reserve_cash": [0.0, 0.1]},
    )

    assert len(result.table) == 6
    assert normalize_calls == 0


def test_single_strategy_parameter_sweep_reuses_loaded_data_and_saves_ranked_table(
    tmp_path: Path, monkeypatch
) -> None:
    data_root = tmp_path / "market" / "daily"
    output_dir = tmp_path / "runs" / "single-sweep"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    original_load = experiment_module.MarketDataRepository.load_backtest_data
    load_calls = 0
    generate_calls = 0

    class EmptySingleStrategy:
        name = "trend_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            nonlocal generate_calls
            generate_calls += 1
            return pd.DataFrame(columns=ORDER_COLUMNS)

    def spy_load(self, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return original_load(self, **kwargs)

    monkeypatch.setattr(experiment_module.MarketDataRepository, "load_backtest_data", spy_load)
    monkeypatch.setattr(experiment_module, "create_strategy_for_detector", lambda detector, cfg: EmptySingleStrategy())
    config = SingleStrategyExperimentConfig(
        name="single-sweep",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        strict_data_quality=False,
        output_dir=str(output_dir),
    )

    result = run_single_strategy_parameter_sweep(
        config,
        grid={"fee_rate": [0.0, 0.001], "slippage_bps": [0.0, 5.0]},
        save=True,
    )

    assert load_calls == 1
    assert generate_calls == 1
    assert len(result.table) == 4
    assert {
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_config_hash",
        "case_name",
        "fee_rate",
        "slippage_bps",
        "trade_count",
        "total_return",
        "bars_per_second",
        "order_cache_status",
        "generated_order_count",
        "order_count",
        "strategy_signal_count",
        "data_weighted_coverage_ratio",
        "filtered_limit_open_count",
    }.issubset(result.table.columns)
    assert result.table["sweep_rank"].tolist() == [1, 2, 3, 4]
    assert result.table["case_config_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert result.table["case_config_hash"].is_unique
    assert result.table.columns[0] == "sweep_rank"
    assert result.table.columns[1] == "pareto_rank"
    assert result.table.columns[2] == "is_pareto_efficient"
    assert result.table.columns[3] == "case_config_hash"
    assert result.table["order_cache_status"].tolist() == ["miss", "hit", "hit", "hit"]
    assert (output_dir / "sweep.csv").exists()
    assert (output_dir / "config.json").exists()
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config["name"] == "single-sweep"
    assert saved_config["sweep_grid"] == {"fee_rate": [0.0, 0.001], "slippage_bps": [0.0, 5.0]}
    saved_sweep = pd.read_csv(output_dir / "sweep.csv")
    assert {
        "sweep_rank",
        "pareto_rank",
        "is_pareto_efficient",
        "case_config_hash",
        "fee_rate",
        "slippage_bps",
        "order_cache_status",
        "generated_order_count",
        "data_weighted_coverage_ratio",
        "filtered_limit_open_count",
    }.issubset(saved_sweep.columns)
    assert saved_sweep["sweep_rank"].tolist() == [1, 2, 3, 4]
    assert saved_sweep["case_config_hash"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_sweep_table_ranking_uses_deterministic_tie_breaks() -> None:
    table = pd.DataFrame(
        {
            "case_name": ["case-b", "case-a", "case-c"],
            "total_return": [0.1, 0.1, 0.2],
            "max_drawdown": [-0.02, -0.02, -0.10],
            "trade_count": [3, 3, 1],
        }
    )

    ranked = experiment_module._rank_sweep_table(table)

    assert ranked["sweep_rank"].tolist() == [1, 2, 3]
    assert ranked["case_name"].tolist() == ["case-c", "case-a", "case-b"]
    assert ranked.columns[0] == "sweep_rank"


def test_sweep_table_ranking_reports_pareto_fronts() -> None:
    table = pd.DataFrame(
        {
            "case_name": ["case-a", "case-b", "case-c", "case-d"],
            "total_return": [0.10, 0.12, 0.08, 0.06],
            "max_drawdown": [-0.05, -0.10, -0.06, -0.12],
            "ulcer_index": [0.03, 0.08, 0.04, 0.09],
            "trade_count": [10, 12, 8, 4],
        }
    )

    ranked = experiment_module._rank_sweep_table(table)

    by_case = ranked.set_index("case_name")
    assert ranked["case_name"].tolist() == ["case-b", "case-a", "case-c", "case-d"]
    assert by_case.loc["case-a", "pareto_rank"] == 1
    assert by_case.loc["case-b", "pareto_rank"] == 1
    assert by_case.loc["case-c", "pareto_rank"] == 2
    assert by_case.loc["case-d", "pareto_rank"] == 3
    assert bool(by_case.loc["case-a", "is_pareto_efficient"]) is True
    assert bool(by_case.loc["case-c", "is_pareto_efficient"]) is False


def test_sweep_case_config_hash_is_stable_and_changes_with_config() -> None:
    base = PortfolioExperimentConfig(
        name="portfolio-sweep",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        strategy_priority={"b": 2, "a": 1},
    )
    same = PortfolioExperimentConfig(
        name="portfolio-sweep",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        strategy_priority={"a": 1, "b": 2},
    )
    changed = PortfolioExperimentConfig(
        name="portfolio-sweep",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        risk_reward=2.5,
        strategy_priority={"a": 1, "b": 2},
    )

    assert experiment_module._case_config_hash(base) == experiment_module._case_config_hash(same)
    assert experiment_module._case_config_hash(base) != experiment_module._case_config_hash(changed)


def test_single_strategy_parameter_sweep_reuses_orders_when_disabled_detector_params_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    generate_calls = 0

    class EmptySingleStrategy:
        name = "trend_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            nonlocal generate_calls
            generate_calls += 1
            return pd.DataFrame(columns=ORDER_COLUMNS)

    monkeypatch.setattr(experiment_module, "create_strategy_for_detector", lambda detector, cfg: EmptySingleStrategy())
    config = SingleStrategyExperimentConfig(
        name="single-sweep-disabled-detector-cache",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        strict_data_quality=False,
    )

    result = run_single_strategy_parameter_sweep(
        config,
        grid={"channel_method": ["regression", "swing"], "range_min_score": [0.6, 0.9]},
    )

    assert generate_calls == 1
    by_case = result.table.sort_values("case_name")
    assert by_case["order_cache_status"].tolist() == ["miss", "hit", "hit", "hit"]


def test_single_strategy_parameter_sweep_does_not_reuse_orders_when_higher_context_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=bars)
    generate_calls = 0

    class EmptySingleStrategy:
        name = "range_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            nonlocal generate_calls
            generate_calls += 1
            return pd.DataFrame(columns=ORDER_COLUMNS)

    monkeypatch.setattr(experiment_module, "create_strategy_for_detector", lambda detector, cfg: EmptySingleStrategy())
    config = SingleStrategyExperimentConfig(
        name="single-sweep-mtf-context-cache",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        higher_timeframe="60m",
        start="2026-05-25",
        end="2026-05-25",
        detector="range",
        strict_data_quality=False,
    )

    result = run_single_strategy_parameter_sweep(config, grid={"trend_lookback": [5, 8]})

    assert generate_calls == 2
    by_case = result.table.sort_values("case_name")
    assert by_case["order_cache_status"].tolist() == ["miss", "miss"]


def test_single_strategy_parameter_sweep_rejects_data_scope_grid_fields(tmp_path: Path, monkeypatch) -> None:
    def fail_load(self, **kwargs):
        raise AssertionError("数据范围字段非法时不应加载行情")

    monkeypatch.setattr(experiment_module.MarketDataRepository, "load_backtest_data", fail_load)
    config = SingleStrategyExperimentConfig(
        name="single-sweep-data-scope",
        data_root=str(tmp_path / "market" / "daily"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )

    with pytest.raises(ValueError, match="不能在同一次 sweep 中改变数据范围字段.*symbols"):
        run_single_strategy_parameter_sweep(config, grid={"symbols": [("000001.SZ",), ("000002.SZ",)]})


def test_single_strategy_experiment_uses_one_detector_without_portfolio_layer(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    output_dir = tmp_path / "runs" / "single-trend"
    close = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 10.9, 10.8, 11.1, 11.5, 11.9, 12.4, 12.9]
    rows: list[dict[str, object]] = []
    for index, value in enumerate(close):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": value - (0.12 if index not in {8, 9} else 0.35),
                "high": value + (0.18 if index not in {8, 9} else 0.08),
                "low": value - (0.18 if index not in {8, 9} else 0.62),
                "close": value,
                "volume": 1000.0 + index * 25.0,
                "amount": value * (1000.0 + index * 25.0),
            }
        )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))

    def fail_if_portfolio_called(*args, **kwargs):
        raise AssertionError("single strategy experiment must not call portfolio backtest")

    monkeypatch.setattr(experiment_module, "run_portfolio_backtest", fail_if_portfolio_called)
    config = SingleStrategyExperimentConfig(
        name="single-trend",
        data_root=str(data_root),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        risk_reward=1.0,
        max_holding_bars=4,
        max_actual_risk_pct=0.08,
        max_chase_pct=0.08,
        min_coverage_ratio=0.1,
        strict_data_quality=False,
        trend_lookback=5,
        trend_min_score=0.2,
        channel_method="swing",
        reversal_old_extreme_tolerance_pct=0.03,
        reversal_require_old_extreme_test=False,
        reversal_require_structure_confirmation=False,
        output_dir=str(output_dir),
    )

    result = run_single_strategy_experiment(config, save=True)

    assert result.config.detector == "trend"
    assert result.elapsed_seconds > 0
    assert result.backtest.stats["trade_count"] >= 1
    assert result.backtest.stats["data_audit_row_count"] == 1.0
    assert result.backtest.stats["data_weighted_coverage_ratio"] == pytest.approx(1.0)
    assert result.backtest.stats["limit_filter_audit_row_count"] == 1.0
    assert result.backtest.stats["limit_filter_filtered_days"] == 0.0
    assert "date" in result.backtest.equity_curve.columns
    assert result.backtest.trades["detector_name"].eq("trend").all()
    assert result.backtest.trades["strategy_name"].eq("trend_signal_bar").all()
    assert result.data_coverage["status"].tolist() == ["ok"]
    assert result.strategy_stats.set_index("strategy_name").loc["trend_signal_bar", "trade_count"] == result.backtest.stats[
        "trade_count"
    ]
    assert not result.symbol_stats.empty
    assert not result.side_stats.empty
    assert not result.exit_reason_stats.empty
    assert not result.monthly_returns.empty
    assert (output_dir / "config.json").exists()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "equity_curve.csv").exists()
    assert (output_dir / "data_coverage.csv").exists()
    assert (output_dir / "strategy_stats.csv").exists()
    assert (output_dir / "symbol_stats.csv").exists()
    assert (output_dir / "side_stats.csv").exists()
    assert (output_dir / "exit_reason_stats.csv").exists()
    assert (output_dir / "monthly_returns.csv").exists()
    saved_config = json.loads((output_dir / "config.json").read_text())
    saved_stats = json.loads((output_dir / "stats.json").read_text())
    assert saved_config["detector"] == "trend"
    assert saved_config["max_actual_risk_pct"] == 0.08
    assert saved_config["max_chase_pct"] == 0.08
    assert saved_config["min_coverage_ratio"] == 0.1
    assert saved_config["channel_method"] == "swing"
    assert saved_config["reversal_old_extreme_tolerance_pct"] == 0.03
    assert saved_config["reversal_require_old_extreme_test"] is False
    assert saved_config["reversal_require_structure_confirmation"] is False
    assert saved_stats["elapsed_seconds"] == pytest.approx(result.elapsed_seconds)
    assert saved_stats["data_audit_row_count"] == 1.0
    assert saved_stats["data_weighted_coverage_ratio"] == pytest.approx(1.0)
    assert saved_stats["data_missing_rows"] == 0.0
    assert saved_stats["limit_filter_audit_row_count"] == 1.0
    assert saved_stats["limit_filter_filtered_days"] == 0.0


def test_single_strategy_experiment_builds_strategy_without_default_suite(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00"]),
            "stock_code": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2],
            "high": [10.3, 10.5],
            "low": [9.8, 10.0],
            "close": [10.2, 10.4],
            "volume": [1000.0, 1100.0],
            "amount": [10200.0, 11440.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    captured: dict[str, object] = {}

    class EmptySingleStrategy:
        name = "range_signal_bar"

        def generate_orders(self, bars: pd.DataFrame, *, timeframe: str = "") -> pd.DataFrame:
            return pd.DataFrame(columns=ORDER_COLUMNS)

    def fail_default_suite(*args, **kwargs):
        raise AssertionError("single strategy experiment must not create a default suite")

    def fake_strategy_factory(detector: str, cfg) -> EmptySingleStrategy:
        captured["detector"] = detector
        captured["enabled"] = cfg.enabled
        return EmptySingleStrategy()

    monkeypatch.setattr(experiment_module, "create_default_strategy_suite", fail_default_suite)
    monkeypatch.setattr(experiment_module, "create_strategy_for_detector", fake_strategy_factory)

    result = run_single_strategy_experiment(
        SingleStrategyExperimentConfig(
            name="single-no-suite",
            data_root=str(data_root),
            symbols=("000001.SZ",),
            timeframe="30m",
            start="2026-05-25",
            end="2026-05-25",
            detector="range",
            strict_data_quality=False,
        )
    )

    assert captured == {"detector": "range", "enabled": ("range",)}
    assert result.backtest.trades.empty
