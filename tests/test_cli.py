from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd

from trending_winning import cli as cli_module
from trending_winning.backtest import experiment as experiment_module
from trending_winning.cli import main
from trending_winning.data.repository import write_local_bars


def test_cli_tdx_doctor_prints_diagnostic_table(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_diagnose_tdx_source(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "timeframe": ["30m"],
                "tdx_period": ["30m"],
                "status": ["ok"],
                "rows": [2],
                "symbols": ["000001.SZ"],
                "start": [pd.Timestamp("2026-05-25 10:30:00")],
                "end": [pd.Timestamp("2026-05-25 11:30:00")],
                "message": ["TDX 样本请求成功。"],
            }
        )

    monkeypatch.setattr(cli_module, "diagnose_tdx_source", fake_diagnose_tdx_source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "tdx-doctor",
            "--symbols",
            "000001.SZ",
            "--timeframes",
            "30m",
            "--start",
            "2026-05-25 09:30:00",
            "--end",
            "2026-05-25 15:00:00",
            "--tdx-path",
            "/tmp/tdx/PYPlugins/user",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "TDX 样本请求成功" in out
    assert captured["symbols"] == ("000001.SZ",)
    assert captured["timeframes"] == ("30m",)
    assert captured["tqcenter_path"] == "/tmp/tdx/PYPlugins/user"


def test_cli_portfolio_backtest_runs_on_local_bars(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00", "2026-05-25 10:00:00", "2026-05-25 10:30:00"]),
            "stock_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "open": [10.0, 10.2, 10.4],
            "high": [10.3, 10.5, 10.7],
            "low": [9.8, 10.0, 10.2],
            "close": [10.2, 10.4, 10.6],
            "volume": [1000.0, 1100.0, 1200.0],
            "amount": [10200.0, 11440.0, 12720.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-backtest",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detectors",
            "trend,range",
            "--output-dir",
            str(tmp_path / "cli-run"),
            "--capital-per-trade",
            "0.25",
            "--reserve-cash",
            "0.1",
            "--allow-same-symbol-overlap",
            "--strategy-priority",
            "trend_signal_bar=1",
            "--short-margin-rate",
            "1.5",
            "--intrabar-exit-policy",
            "optimistic",
            "--trend-lookback",
            "7",
            "--trend-min-score",
            "0.4",
            "--trend-h2-min-pullback-legs",
            "3",
            "--range-lookback",
            "8",
            "--channel-lookback",
            "9",
            "--channel-sigma-multiple",
            "1.6",
            "--reversal-lookback",
            "10",
            "--reversal-old-extreme-tolerance-pct",
            "0.02",
            "--disable-reversal-old-extreme-test",
            "--disable-reversal-structure-confirmation",
            "--strategy-capital-limit",
            "trend_signal_bar=0.6",
            "--sector-capital-limit",
            "银行=0.5",
            "--symbol-sector-map",
            "000001.SZ=银行",
            "--benchmark",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "gross_exposure" in out
    assert "open_positions" in out
    assert (tmp_path / "cli-run" / "config.json").exists()
    assert (tmp_path / "cli-run" / "equity_curve.csv").exists()
    assert (tmp_path / "cli-run" / "benchmark.json").exists()
    saved_config = json.loads((tmp_path / "cli-run" / "config.json").read_text())
    assert saved_config["short_margin_rate"] == 1.5
    assert saved_config["capital_per_trade"] == 0.25
    assert saved_config["reserve_cash"] == 0.1
    assert saved_config["allow_same_symbol_overlap"] is True
    assert saved_config["strategy_priority"] == {"trend_signal_bar": 1}
    assert saved_config["intrabar_exit_policy"] == "optimistic"
    assert saved_config["trend_lookback"] == 7
    assert saved_config["trend_min_score"] == 0.4
    assert saved_config["trend_h2_min_pullback_legs"] == 3
    assert saved_config["range_lookback"] == 8
    assert saved_config["channel_lookback"] == 9
    assert saved_config["channel_sigma_multiple"] == 1.6
    assert saved_config["reversal_lookback"] == 10
    assert saved_config["reversal_old_extreme_tolerance_pct"] == 0.02
    assert saved_config["reversal_require_old_extreme_test"] is False
    assert saved_config["reversal_require_structure_confirmation"] is False
    assert saved_config["strategy_capital_limit"] == {"trend_signal_bar": 0.6}
    assert saved_config["sector_capital_limit"] == {"银行": 0.5}
    assert saved_config["symbol_sector_map"] == {"000001.SZ": "银行"}


def test_cli_portfolio_backtest_benchmark_reuses_single_backtest_run(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    rows: list[dict[str, object]] = []
    for symbol in ("000001.SZ", "000002.SZ"):
        for index in range(8):
            close = 10.0 + index * 0.06
            rows.append(
                {
                    "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                    "stock_code": symbol,
                    "open": close - 0.05,
                    "high": close + 0.15,
                    "low": close - 0.15,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))
    original_backtest = experiment_module.run_portfolio_backtest
    backtest_calls = 0

    def spy_backtest(*args, **kwargs):
        nonlocal backtest_calls
        backtest_calls += 1
        return original_backtest(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "run_portfolio_backtest", spy_backtest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-backtest",
            "--symbols",
            "000001.SZ,000002.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detectors",
            "trend,range",
            "--output-dir",
            str(tmp_path / "bench-run"),
            "--benchmark",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert backtest_calls == 1
    assert "bars_per_second" in out
    assert (tmp_path / "bench-run" / "benchmark.json").exists()


def test_cli_portfolio_backtest_passes_higher_timeframe_gate_config(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run(config, *, save=False):
        captured["config"] = config
        captured["save"] = save
        trades = pd.DataFrame(columns=["strategy_name", "stock_code", "return_pct", "holding_bars"])
        return experiment_module.PortfolioExperimentResult(
            config=config,
            backtest=experiment_module.BacktestResult(
                trades=trades,
                equity_curve=pd.DataFrame(),
                stats={"trade_count": 0.0},
            ),
            input_bar_count=0,
            filtered_limit_open_count=0,
            data_coverage=pd.DataFrame(),
            strategy_stats=pd.DataFrame(),
            symbol_stats=pd.DataFrame(),
            side_stats=pd.DataFrame(),
            exit_reason_stats=pd.DataFrame(),
            monthly_returns=pd.DataFrame(),
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr("trending_winning.cli.run_portfolio_experiment", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-backtest",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "15m",
            "--higher-timeframe",
            "60m",
            "--higher-timeframe-max-age-minutes",
            "90",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            "/tmp/trend-data",
            "--detectors",
            "trend",
        ],
    )

    main()

    out = capsys.readouterr().out
    config = captured["config"]
    assert "trade_count" in out
    assert config.higher_timeframe == "60m"
    assert config.higher_timeframe_max_age_minutes == 90
    assert captured["save"] is False


def test_cli_audit_data_reports_missing_and_existing_symbols(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00"]),
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "high": [10.3],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
        }
    )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "audit-data",
            "--symbols",
            "000001.SZ,000002.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "000001.SZ" in out
    assert "ok" in out
    assert "000002.SZ" in out
    assert "missing_file" in out


def test_cli_audit_data_can_include_higher_timeframe(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    bars_30m = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25 09:30:00"]),
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "high": [10.3],
            "low": [9.8],
            "close": [10.2],
            "volume": [1000.0],
            "amount": [10200.0],
        }
    )
    bars_60m = bars_30m.assign(date=pd.to_datetime(["2026-05-25 10:30:00"]), close=[10.4], amount=[10400.0])
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=bars_30m)
    write_local_bars(data_root=data_root, timeframe="60m", adjust="qfq", bars=bars_60m)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "audit-data",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--higher-timeframe",
            "60m",
            "--higher-timeframe-max-age-minutes",
            "120",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "30m" in out
    assert "60m" in out
    assert sum(line.lstrip().startswith("000001.SZ") for line in out.splitlines()) == 2


def test_cli_prepare_data_calls_repository_prepare_from_tdx(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    captured: dict[str, object] = {}

    def fake_prepare(self, **kwargs):
        captured["data_root"] = self.data_root
        captured["adjust"] = self.adjust
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "timeframe": ["60m"],
                "action": ["fetched"],
                "before_status": ["missing_file"],
                "after_status": ["ok"],
            }
        )

    monkeypatch.setattr(experiment_module.MarketDataRepository, "prepare_from_tdx", fake_prepare, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "prepare-data",
            "--symbols",
            "000001.SZ",
            "--timeframes",
            "30m,60m",
            "--start",
            "2026-05-25 09:30:00",
            "--end",
            "2026-05-25 15:00:00",
            "--adjust",
            "qfq",
            "--data-root",
            str(data_root),
            "--tdx-path",
            "/tmp/tdx/PYPlugins/user",
            "--min-coverage-ratio",
            "0.95",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "fetched" in out
    assert captured["data_root"] == data_root
    assert captured["adjust"] == "qfq"
    assert captured["symbols"] == ("000001.SZ",)
    assert captured["timeframes"] == ("30m", "60m")
    assert captured["tqcenter_path"] == "/tmp/tdx/PYPlugins/user"
    assert captured["min_coverage_ratio"] == 0.95


def test_cli_plan_data_calls_repository_plan_from_tdx(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    captured: dict[str, object] = {}

    def fake_plan(self, **kwargs):
        captured["data_root"] = self.data_root
        captured["adjust"] = self.adjust
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "stock_code": ["000001.SZ"],
                "timeframe": ["60m"],
                "action": ["fetch"],
                "reason": ["missing_file"],
            }
        )

    monkeypatch.setattr(experiment_module.MarketDataRepository, "plan_from_tdx", fake_plan, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "plan-data",
            "--symbols",
            "000001.SZ",
            "--timeframes",
            "5m,15m,30m,60m",
            "--start",
            "2026-05-25 09:30:00",
            "--end",
            "2026-05-25 15:00:00",
            "--adjust",
            "qfq",
            "--data-root",
            str(data_root),
            "--min-coverage-ratio",
            "0.95",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "fetch" in out
    assert captured["data_root"] == data_root
    assert captured["adjust"] == "qfq"
    assert captured["symbols"] == ("000001.SZ",)
    assert captured["timeframes"] == ("5m", "15m", "30m", "60m")
    assert captured["min_coverage_ratio"] == 0.95


def test_cli_portfolio_backtest_can_allow_bad_data_explicitly(tmp_path: Path, monkeypatch, capsys) -> None:
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
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-backtest",
            "--symbols",
            "000003.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "trade_count" in out


def test_cli_single_strategy_backtest_saves_without_portfolio_outputs(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
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
    output_dir = tmp_path / "single-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "single-backtest",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detector",
            "trend",
            "--risk-reward",
            "1.0",
            "--max-holding-bars",
            "4",
            "--max-actual-risk-pct",
            "0.08",
            "--max-chase-pct",
            "0.08",
            "--min-coverage-ratio",
            "0.1",
            "--fee-rate",
            "0.0003",
            "--slippage-bps",
            "5",
            "--initial-equity",
            "2",
            "--trend-lookback",
            "5",
            "--trend-min-score",
            "0.2",
            "--trend-strong-close-pos",
            "0.7",
            "--trend-min-body-ratio",
            "0.55",
            "--trend-pullback-lookback",
            "4",
            "--trend-h2-min-pullback-legs",
            "3",
            "--range-middle-low",
            "0.2",
            "--range-middle-high",
            "0.8",
            "--range-false-break-buffer",
            "0.01",
            "--range-strong-close-pos",
            "0.7",
            "--range-min-score",
            "0.9",
            "--channel-method",
            "swing",
            "--channel-break-buffer",
            "0.02",
            "--channel-swing-left-bars",
            "3",
            "--channel-swing-right-bars",
            "4",
            "--reversal-strong-close-pos",
            "0.7",
            "--reversal-min-body-ratio",
            "0.5",
            "--reversal-old-extreme-tolerance-pct",
            "0.03",
            "--disable-reversal-old-extreme-test",
            "--disable-reversal-structure-confirmation",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "trade_count" in out
    assert (output_dir / "config.json").exists()
    assert (output_dir / "trades.csv").exists()
    assert (output_dir / "order_decisions.csv").exists()
    assert (output_dir / "strategy_stats.csv").exists()
    assert (output_dir / "symbol_stats.csv").exists()
    assert (output_dir / "side_stats.csv").exists()
    assert (output_dir / "exit_reason_stats.csv").exists()
    assert (output_dir / "monthly_returns.csv").exists()
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert saved_config["detector"] == "trend"
    assert saved_config["max_actual_risk_pct"] == 0.08
    assert saved_config["max_chase_pct"] == 0.08
    assert saved_config["min_coverage_ratio"] == 0.1
    assert saved_config["fee_rate"] == 0.0003
    assert saved_config["slippage_bps"] == 5.0
    assert saved_config["initial_equity"] == 2.0
    assert saved_config["trend_strong_close_pos"] == 0.7
    assert saved_config["trend_min_body_ratio"] == 0.55
    assert saved_config["trend_pullback_lookback"] == 4
    assert saved_config["trend_h2_min_pullback_legs"] == 3
    assert saved_config["range_middle_low"] == 0.2
    assert saved_config["range_middle_high"] == 0.8
    assert saved_config["range_false_break_buffer"] == 0.01
    assert saved_config["range_strong_close_pos"] == 0.7
    assert saved_config["range_min_score"] == 0.9
    assert saved_config["channel_method"] == "swing"
    assert saved_config["channel_break_buffer"] == 0.02
    assert saved_config["channel_swing_left_bars"] == 3
    assert saved_config["channel_swing_right_bars"] == 4
    assert saved_config["reversal_strong_close_pos"] == 0.7
    assert saved_config["reversal_min_body_ratio"] == 0.5
    assert saved_config["reversal_old_extreme_tolerance_pct"] == 0.03
    assert saved_config["reversal_require_old_extreme_test"] is False
    assert saved_config["reversal_require_structure_confirmation"] is False


def test_cli_portfolio_sweep_saves_parameter_table(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    rows: list[dict[str, object]] = []
    for index in range(10):
        close = 10.0 + index * 0.08
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.05,
                "high": close + 0.20,
                "low": close - 0.15,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))
    output_dir = tmp_path / "sweep-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-sweep",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detectors",
            "trend,range",
            "--risk-rewards",
            "1.5,2.0",
            "--max-holding-bars-list",
            "3,5",
            "--capital-per-trade",
            "0.2",
            "--reserve-cash",
            "0.1",
            "--allow-same-symbol-overlap",
            "--strategy-priority",
            "trend_signal_bar=1",
            "--strategy-capital-limit",
            "trend_signal_bar=0.6",
            "--sector-capital-limit",
            "银行=0.5",
            "--symbol-sector-map",
            "000001.SZ=银行",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "sweep.csv" in out
    saved = pd.read_csv(output_dir / "sweep.csv")
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert len(saved) == 4
    assert {"risk_reward", "max_holding_bars", "total_return"}.issubset(saved.columns)
    assert saved_config["capital_per_trade"] == 0.2
    assert saved_config["reserve_cash"] == 0.1
    assert saved_config["allow_same_symbol_overlap"] is True
    assert saved_config["strategy_priority"] == {"trend_signal_bar": 1}
    assert saved_config["strategy_capital_limit"] == {"trend_signal_bar": 0.6}
    assert saved_config["sector_capital_limit"] == {"银行": 0.5}
    assert saved_config["symbol_sector_map"] == {"000001.SZ": "银行"}


def test_cli_single_sweep_saves_parameter_table(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "market" / "daily"
    rows: list[dict[str, object]] = []
    for index in range(10):
        close = 10.0 + index * 0.08
        rows.append(
            {
                "date": pd.Timestamp("2026-05-25 09:30:00") + pd.Timedelta(minutes=30 * index),
                "stock_code": "000001.SZ",
                "open": close - 0.05,
                "high": close + 0.20,
                "low": close - 0.15,
                "close": close,
                "volume": 1000.0,
                "amount": close * 1000.0,
            }
        )
    write_local_bars(data_root=data_root, timeframe="30m", adjust="qfq", bars=pd.DataFrame(rows))
    output_dir = tmp_path / "single-sweep-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "single-sweep",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detector",
            "trend",
            "--risk-rewards",
            "1.5,2.0",
            "--max-holding-bars-list",
            "3,5",
            "--trend-min-scores",
            "0.2,0.4",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    out = capsys.readouterr().out
    assert "sweep.csv" in out
    saved = pd.read_csv(output_dir / "sweep.csv")
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert len(saved) == 8
    assert {"detector", "risk_reward", "max_holding_bars", "trend_min_score", "total_return"}.issubset(saved.columns)
    assert saved_config["detector"] == "trend"
    assert saved_config["sweep_grid"] == {
        "risk_reward": [1.5, 2.0],
        "max_holding_bars": [3, 5],
        "trend_min_score": [0.2, 0.4],
    }


def test_cli_single_sweep_accepts_generic_grid_fields(tmp_path: Path, monkeypatch, capsys) -> None:
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
    output_dir = tmp_path / "single-generic-grid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "single-sweep",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detector",
            "range",
            "--risk-rewards",
            "1.5",
            "--max-holding-bars-list",
            "3",
            "--grid",
            "range_min_score=0.7,0.9",
            "--grid",
            "fee_rate=0,0.001",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    capsys.readouterr()
    saved = pd.read_csv(output_dir / "sweep.csv")
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert len(saved) == 4
    assert {"range_min_score", "fee_rate", "risk_reward", "max_holding_bars"}.issubset(saved.columns)
    assert saved_config["sweep_grid"] == {
        "risk_reward": [1.5],
        "max_holding_bars": [3],
        "range_min_score": [0.7, 0.9],
        "fee_rate": [0.0, 0.001],
    }


def test_cli_portfolio_sweep_accepts_generic_grid_fields(tmp_path: Path, monkeypatch, capsys) -> None:
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
    output_dir = tmp_path / "portfolio-generic-grid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-sweep",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detectors",
            "trend",
            "--risk-rewards",
            "2.0",
            "--max-holding-bars-list",
            "3",
            "--grid",
            "reserve_cash=0,0.1",
            "--grid",
            "allow_same_symbol_overlap=true,false",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    capsys.readouterr()
    saved = pd.read_csv(output_dir / "sweep.csv")
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert len(saved) == 4
    assert {"reserve_cash", "allow_same_symbol_overlap", "risk_reward", "max_holding_bars"}.issubset(saved.columns)
    assert saved_config["sweep_grid"] == {
        "risk_reward": [2.0],
        "max_holding_bars": [3],
        "max_open_positions": [5],
        "reserve_cash": [0.0, 0.1],
        "allow_same_symbol_overlap": [True, False],
    }


def test_cli_portfolio_sweep_accepts_mapping_grid_fields(tmp_path: Path, monkeypatch, capsys) -> None:
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
    output_dir = tmp_path / "portfolio-mapping-grid"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "trending-winning",
            "portfolio-sweep",
            "--symbols",
            "000001.SZ",
            "--timeframe",
            "30m",
            "--start",
            "2026-05-25",
            "--end",
            "2026-05-25",
            "--data-root",
            str(data_root),
            "--allow-bad-data",
            "--detectors",
            "trend",
            "--risk-rewards",
            "2.0",
            "--max-holding-bars-list",
            "3",
            "--grid",
            "strategy_capital_limit=trend_signal_bar=0.4;trend_signal_bar=0.8",
            "--grid",
            "symbol_sector_map=000001.SZ=银行;000001.SZ=券商",
            "--output-dir",
            str(output_dir),
        ],
    )

    main()

    capsys.readouterr()
    saved = pd.read_csv(output_dir / "sweep.csv")
    saved_config = json.loads((output_dir / "config.json").read_text())
    assert len(saved) == 4
    assert saved_config["sweep_grid"]["strategy_capital_limit"] == [
        {"trend_signal_bar": 0.4},
        {"trend_signal_bar": 0.8},
    ]
    assert saved_config["sweep_grid"]["symbol_sector_map"] == [
        {"000001.SZ": "银行"},
        {"000001.SZ": "券商"},
    ]
