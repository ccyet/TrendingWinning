from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from trending_winning.backtest.models import BacktestResult
from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport,
    PortfolioExperimentConfig,
    PortfolioExperimentResult,
    SingleStrategyExperimentConfig,
    SingleStrategyExperimentResult,
    SingleStrategySweepResult,
)


def test_experiment_output_imports_without_experiment_runner_and_saves_sweep(tmp_path) -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_output import save_single_strategy_sweep

    config = SingleStrategyExperimentConfig(
        name="single-sweep",
        data_root="/data",
        output_dir=str(tmp_path / "runs"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    result = SingleStrategySweepResult(
        config=config,
        grid={"risk_reward": [2.0]},
        table=pd.DataFrame(),
        data_coverage=pd.DataFrame(),
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
    )

    output_dir = save_single_strategy_sweep(result)

    saved_config = json.loads((output_dir / "config.json").read_text())
    saved_cases = [json.loads(line) for line in (output_dir / "case_configs.jsonl").read_text().splitlines()]
    assert "trending_winning.backtest.experiment" not in sys.modules
    assert saved_config["sweep_grid"] == {"risk_reward": [2.0]}
    assert saved_cases[0]["case_name"] == "single-sweep-001"
    assert (output_dir / "parameter_summary.csv").exists()
    assert (output_dir / "case_diagnostics.csv").exists()
    assert (output_dir / "symbol_metadata.csv").exists()
    manifest = pd.read_csv(output_dir / "artifact_manifest.csv")
    assert manifest.columns.tolist() == ["file_name", "category", "priority", "question", "description"]
    assert manifest.set_index("file_name").loc["sweep.csv", "category"] == "参数遍历"
    assert "先筛选参数组" in manifest.set_index("file_name").loc["sweep.csv", "question"]
    assert manifest.set_index("file_name").loc["case_configs.jsonl", "priority"] == 2


def test_save_single_strategy_sweep_writes_case_diagnostics(tmp_path) -> None:
    from trending_winning.backtest.experiment_cases import case_config_hash
    from trending_winning.backtest.experiment_output import save_single_strategy_sweep

    config = SingleStrategyExperimentConfig(
        name="single-sweep-diagnostics",
        data_root="/data",
        output_dir=str(tmp_path / "runs"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    case_hash = case_config_hash(config)
    result = SingleStrategySweepResult(
        config=config,
        grid={"risk_reward": [2.0]},
        table=pd.DataFrame(
            {
                "case_name": ["single-sweep-diagnostics-001"],
                "case_config_hash": [case_hash],
                "sweep_rank": [1],
                "pareto_rank": [1],
                "is_pareto_efficient": [True],
                "trade_count": [0.0],
                "order_count": [0.0],
            }
        ),
        data_coverage=pd.DataFrame(),
        input_bar_count=10,
        filtered_limit_open_count=1,
        elapsed_seconds=0.1,
    )

    output_dir = save_single_strategy_sweep(result)

    saved = pd.read_csv(output_dir / "case_diagnostics.csv")
    assert {"sweep_rank", "case_name", "check", "status"}.issubset(saved.columns)
    assert saved.loc[saved["check"].eq("交易样本"), "status"].iloc[0] == "失败"


def test_save_single_strategy_sweep_writes_html_overview_report(tmp_path) -> None:
    from trending_winning.backtest.experiment_cases import case_config_hash, sweep_variants
    from trending_winning.backtest.experiment_output import save_single_strategy_sweep

    config = SingleStrategyExperimentConfig(
        name="single-sweep-html",
        data_root="/data",
        output_dir=str(tmp_path / "single-sweep-html"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    grid = {"risk_reward": [1.5, 2.0], "trend_min_score": [0.8]}
    variants = sweep_variants(config, grid)
    result = SingleStrategySweepResult(
        config=config,
        grid=grid,
        table=pd.DataFrame(
            {
                "sweep_rank": [1, 2],
                "pareto_rank": [1, 2],
                "is_pareto_efficient": [True, False],
                "risk_adjusted_rank": [2, 1],
                "risk_adjusted_score": [72.0, 86.5],
                "case_name": ["single-sweep-html-001", "single-sweep-html-002"],
                "case_config_hash": [case_config_hash(variant) for variant in variants],
                "risk_reward": [1.5, 2.0],
                "trend_min_score": [0.8, 0.8],
                "total_return": [0.08, 0.05],
                "max_drawdown": [-0.06, -0.03],
                "trade_count": [34.0, 22.0],
                "win_rate": [0.56, 0.62],
                "acceptance_rate": [0.48, 0.41],
                "diagnostic_primary_issue": ["", "交易样本"],
            }
        ),
        data_coverage=pd.DataFrame(),
        input_bar_count=200,
        filtered_limit_open_count=1,
        elapsed_seconds=1.25,
    )

    output_dir = save_single_strategy_sweep(result)

    html = (output_dir / "sweep_report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "single-sweep-html" in html
    assert "参数遍历总览" in html
    assert "最优参数组" in html
    assert "风险质量候选" in html
    assert "Pareto候选" in html
    assert "参数影响" in html
    assert "诊断概览" in html
    assert "重点产物" in html
    assert 'class="report-table"' in html
    assert "text-align:center" in html
    assert "<th>参数组</th>" in html
    assert "<th>总收益</th>" in html
    assert "<th>最大回撤</th>" in html
    assert "<th>诊断主问题</th>" in html
    assert "single-sweep-html-001" in html
    assert "single-sweep-html-002" in html
    assert "8.00%" in html
    assert "86.50" in html
    assert "parameter_summary.csv" in html
    manifest = pd.read_csv(output_dir / "artifact_manifest.csv").set_index("file_name")
    assert manifest.loc["sweep_report.html", "category"] == "阅读入口"
    assert manifest.loc["sweep_report.html", "priority"] == 0


def test_experiment_runner_reexports_output_functions_for_compatibility() -> None:
    from trending_winning.backtest import experiment
    from trending_winning.backtest.experiment_output import (
        save_portfolio_benchmark,
        save_portfolio_experiment,
        save_portfolio_sweep,
        save_single_strategy_experiment,
        save_single_strategy_sweep,
    )

    assert experiment.save_single_strategy_experiment is save_single_strategy_experiment
    assert experiment.save_portfolio_experiment is save_portfolio_experiment
    assert experiment.save_portfolio_sweep is save_portfolio_sweep
    assert experiment.save_single_strategy_sweep is save_single_strategy_sweep
    assert experiment.save_portfolio_benchmark is save_portfolio_benchmark


def test_save_portfolio_benchmark_writes_strict_json(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_portfolio_benchmark

    config = PortfolioExperimentConfig(
        name="bench",
        data_root="/data",
        output_dir=str(tmp_path / "bench"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
    )
    report = PortfolioBenchmarkReport(
        experiment_name="bench",
        bar_count=10,
        trade_count=2,
        equity_points=3,
        elapsed_seconds=0.5,
        bars_per_second=20.0,
        trades_per_second=4.0,
    )

    output_dir = save_portfolio_benchmark(config, report)

    assert json.loads((output_dir / "benchmark.json").read_text())["bars_per_second"] == 20.0


def test_save_single_strategy_experiment_writes_drawdown_episodes(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-drawdown",
        data_root="/data",
        output_dir=str(tmp_path / "single-drawdown"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-25", periods=5),
            "net_value": [1.0, 1.2, 1.0, 0.9, 1.21],
            "drawdown_net_value": [1.0, 1.2, 1.0, 0.9, 1.21],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=equity,
            stats={"trade_count": 0.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "drawdown_episodes.csv")
    assert saved.loc[0, "depth"] == pytest.approx(0.9 / 1.2 - 1.0)
    assert saved.loc[0, "start_at"] == "2026-05-26 00:00:00"
    assert saved.loc[0, "trough_at"] == "2026-05-28 00:00:00"


def test_save_single_strategy_experiment_writes_data_gap_episodes(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-data-gaps",
        data_root="/data",
        output_dir=str(tmp_path / "single-data-gaps"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    gap_episodes = pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "timeframe": ["30m"],
            "adjust": ["qfq"],
            "gap_no": [1],
            "start_at": [pd.Timestamp("2026-05-25 14:00:00")],
            "end_at": [pd.Timestamp("2026-05-25 14:30:00")],
            "missing_rows": [2],
            "gap_minutes": [60],
            "previous_available_at": [pd.Timestamp("2026-05-25 13:30:00")],
            "next_available_at": [pd.Timestamp("2026-05-25 15:00:00")],
            "requested_start": [pd.Timestamp("2026-05-25 09:30:00")],
            "requested_end": [pd.Timestamp("2026-05-25 15:00:00")],
            "path": ["/data/30m/qfq/000001.SZ.parquet"],
            "status": ["missing_bars"],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"trade_count": 0.0}),
        input_bar_count=0,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        data_gap_episodes=gap_episodes,
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "data_gap_episodes.csv")
    assert saved["stock_code"].tolist() == ["000001.SZ"]
    assert saved["start_at"].tolist() == ["2026-05-25 14:00:00"]
    assert saved["missing_rows"].tolist() == [2]


def test_save_single_strategy_experiment_writes_artifact_manifest(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-manifest",
        data_root="/data",
        output_dir=str(tmp_path / "single-manifest"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"trade_count": 0.0}),
        input_bar_count=0,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    manifest = pd.read_csv(output_dir / "artifact_manifest.csv")
    assert manifest.columns.tolist() == ["file_name", "category", "priority", "question", "description"]
    by_file = manifest.set_index("file_name")
    assert by_file.loc["strategy_space.csv", "category"] == "运行前复核"
    assert by_file.loc["strategy_space.csv", "priority"] == 1
    assert "本次到底启用了哪些策略边界" in by_file.loc["strategy_space.csv", "question"]
    assert by_file.loc["data_gap_episodes.csv", "category"] == "数据质量"
    assert "每段连续缺失 K" in by_file.loc["data_gap_episodes.csv", "description"]
    assert by_file.loc["order_decisions.csv", "category"] == "订单与过滤"


def test_save_single_strategy_experiment_writes_strategy_space_summary(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-strategy-space",
        data_root="/data",
        output_dir=str(tmp_path / "single-strategy-space"),
        symbols=("000001.SZ", "600519.SH"),
        timeframe="30m",
        start="2026-05-01",
        end="2026-05-25",
        detector="trend",
        higher_timeframe="60m",
        higher_timeframe_max_age_minutes=240,
        side_mode="long_only",
        max_actual_risk_pct=0.05,
        max_chase_pct=0.02,
        trailing_take_profit_drawdown_pct=0.015,
        trailing_take_profit_ma_period=20,
        terminal_false_breakout_enabled=True,
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"trade_count": 0.0}),
        input_bar_count=0,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "strategy_space.csv")
    assert saved.columns.tolist() == ["策略空间", "当前设置", "触发与信号", "可能性分类", "边界/输出"]
    assert saved["策略空间"].tolist() == [
        "样本",
        "识别形态",
        "适用空间",
        "信号条件",
        "触发成交",
        "开仓过滤",
        "退出条件",
        "仓位规则",
        "或然分支",
        "复盘输出",
    ]
    joined = " ".join(saved.astype(str).to_numpy().ravel())
    assert "单策略" in joined
    assert "趋势" in joined
    assert "只运行一个识别模块" in joined
    assert "适合趋势延续" in joined
    assert "信号K" in joined
    assert "信号不等于成交" in joined
    assert "入场触发价 = 信号K高点 + tick" in joined
    assert "触发成交条件：多头 high >= 入场触发价，空头 low <= 入场触发价" in joined
    assert "无信号、观察信号、有效信号、有效未触发、触发成交、触发后拒单、持仓冲突、退出完成" in joined
    assert "背景不满足 -> 无信号；背景满足但信号K质量不足 -> 观察信号；信号成立但未穿越挂单价 -> 有效但未触发；穿越后风险不合格 -> 触发后拒单" in joined
    assert "策略空间清单 = 样本空间、标的空间、周期空间、形态空间、参数空间、过滤空间、订单空间、风险空间、持仓空间、执行空间、退出空间、统计空间、失效空间" in joined
    assert "早期顺势、趋势中段回撤、趋势末端衰竭、区间边缘、通道外扩、通道破坏、第一次反转观察、第二次反转确认" in joined
    assert "样本空间、形态空间、过滤空间、执行空间、退出空间、统计空间" in joined
    assert "挂单" in joined
    assert "成交、未触发、方向禁用、追价超限、结构止损风险超限" in joined
    assert "有效信号、有效但未触发、过滤拒单、撮合拒单、持仓冲突" in joined
    assert "仅多" in joined
    assert "大周期方向过滤" in joined
    assert "末端假突破" in joined
    assert "满仓进出" in joined
    assert "候选信号 -> 策略过滤 -> 订单触发 -> 仓位检查 -> 退出" in joined


def test_save_portfolio_experiment_writes_strategy_space_summary(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_portfolio_experiment

    config = PortfolioExperimentConfig(
        name="portfolio-strategy-space",
        data_root="/data",
        output_dir=str(tmp_path / "portfolio-strategy-space"),
        symbols=("000001.SZ", "600519.SH", "300750.SZ"),
        timeframe="15m",
        start="2026-05-01",
        end="2026-05-25",
        detectors=("trend", "channel"),
        side_mode="both",
        max_open_positions=3,
        capital_per_trade=0.3,
        risk_per_trade=0.01,
        reserve_cash=0.1,
        short_margin_rate=2.0,
        strategy_priority={"trend_signal_bar": 1, "channel_signal_bar": 2},
        strategy_capital_limit={"trend_signal_bar": 0.6},
        sector_capital_limit={"新能源": 0.4},
        symbol_sector_map={"300750.SZ": "新能源"},
    )
    result = PortfolioExperimentResult(
        config=config,
        backtest=BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), stats={"trade_count": 0.0}),
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

    output_dir = save_portfolio_experiment(result)

    saved = pd.read_csv(output_dir / "strategy_space.csv")
    assert saved["策略空间"].tolist() == [
        "样本",
        "识别形态",
        "适用空间",
        "信号条件",
        "触发成交",
        "开仓过滤",
        "退出条件",
        "仓位规则",
        "或然分支",
        "复盘输出",
    ]
    joined = " ".join(saved.astype(str).to_numpy().ravel())
    assert "组合策略" in joined
    assert "趋势、通道" in joined
    assert "适合比较多个形态在同一批 K 线里的机会质量" in joined
    assert "信号不等于成交" in joined
    assert "入场触发价 = 信号K高点 + tick" in joined
    assert "触发成交条件：多头 high >= 入场触发价，空头 low <= 入场触发价" in joined
    assert "无信号、观察信号、有效信号、有效未触发、触发成交、触发后拒单、容量/资金拒单、退出完成" in joined
    assert "背景不满足 -> 无信号；背景满足但信号K质量不足 -> 观察信号；信号成立但未穿越挂单价 -> 有效但未触发；穿越后风险不合格 -> 触发后拒单" in joined
    assert "策略空间清单 = 样本空间、标的空间、周期空间、形态空间、参数空间、过滤空间、订单空间、风险空间、持仓空间、执行空间、退出空间、统计空间、失效空间" in joined
    assert "早期顺势、趋势中段回撤、趋势末端衰竭、区间边缘、通道外扩、通道破坏、第一次反转观察、第二次反转确认" in joined
    assert "样本空间、形态空间、过滤空间、执行空间、退出空间、统计空间" in joined
    assert "有效信号、有效但未触发、过滤拒单、撮合拒单、容量/资金拒单" in joined
    assert "资金分配" in joined
    assert "最大持仓 3" in joined
    assert "策略优先级" in joined
    assert "策略资金上限" in joined
    assert "行业资金上限" in joined
    assert "组合净值" in joined
    assert "候选信号 -> 策略过滤 -> 订单触发 -> 组合分配 -> 退出" in joined


def test_save_single_strategy_experiment_writes_drawdown_curve(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-drawdown-curve",
        data_root="/data",
        output_dir=str(tmp_path / "single-drawdown-curve"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-25", "2026-05-26", "2026-05-27"]),
            "net_value": [1.0, 1.0, 1.2],
            "drawdown_net_value": [1.0, 0.8, 1.2],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=equity,
            stats={"trade_count": 0.0},
        ),
        input_bar_count=3,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "drawdown_curve.csv")
    assert saved["point_type"].tolist() == ["settlement", "adverse_price", "settlement", "settlement"]
    assert saved["path_net_value"].tolist() == pytest.approx([1.0, 0.8, 1.0, 1.2])
    assert saved["drawdown"].tolist() == pytest.approx([0.0, -0.2, 0.0, 0.0])


def test_save_single_strategy_experiment_writes_trade_path_distribution(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-path",
        data_root="/data",
        output_dir=str(tmp_path / "single-path"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    trades = pd.DataFrame(
        {
            "return_pct": [3.0, -1.0],
            "holding_bars": [1, 10],
            "r_multiple": [0.8, -1.2],
            "mae_r": [-0.2, -1.1],
            "mfe_r": [1.0, 0.1],
        }
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=trades,
            equity_curve=pd.DataFrame(),
            stats={"trade_count": 2.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "trade_path_distribution.csv")
    assert {"dimension", "bucket", "trade_count", "win_rate", "avg_return"}.issubset(saved.columns)
    assert saved.loc[saved["bucket"].eq("9-16K"), "trade_count"].iloc[0] == 1.0


def test_save_single_strategy_experiment_writes_experiment_diagnostics(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-diagnostics",
        data_root="/data",
        output_dir=str(tmp_path / "single-diagnostics"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            stats={"trade_count": 0.0, "order_count": 0.0},
        ),
        input_bar_count=5,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
    )

    output_dir = save_single_strategy_experiment(result)

    saved = pd.read_csv(output_dir / "experiment_diagnostics.csv")
    assert {"section", "check", "status", "detail"}.issubset(saved.columns)
    assert saved.loc[saved["check"].eq("交易样本"), "status"].iloc[0] == "失败"
    action_plan = pd.read_csv(output_dir / "diagnostic_action_plan.csv")
    assert action_plan["priority"].tolist()[0] == 1
    assert action_plan.loc[0, "check"] == "交易样本"
    assert action_plan.loc[0, "evidence_file"] == "strategy_space.csv; order_decisions.csv"


def test_save_single_strategy_experiment_writes_html_overview_report(tmp_path) -> None:
    from trending_winning.backtest.experiment_output import save_single_strategy_experiment

    config = SingleStrategyExperimentConfig(
        name="single-html-report",
        data_root="/data",
        output_dir=str(tmp_path / "single-html-report"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
        min_coverage_ratio=0.95,
    )
    result = SingleStrategyExperimentResult(
        config=config,
        backtest=BacktestResult(
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame({"trade_no": [0, 1, 2], "net_value": [1.0, 1.08, 1.03]}),
            stats={
                "trade_count": 12.0,
                "order_count": 20.0,
                "win_rate": 0.58,
                "total_return": 0.08,
                "max_drawdown": -0.05,
                "current_drawdown": -0.02,
                "ulcer_index": 0.018,
                "profit_factor": float("inf"),
                "acceptance_rate": 0.55,
                "strategy_filter_rejection_rate": 0.18,
                "primary_rejected_reason": "not_triggered",
                "primary_rejected_reason_count": 5.0,
                "primary_rejected_reason_rate": 0.42,
                "primary_strategy_rejected_reason": "terminal_false_breakout_risk",
                "primary_strategy_rejected_reason_count": 2.0,
                "primary_strategy_rejected_reason_rate": 0.18,
                "exposure_bar_ratio": 0.36,
                "avg_cash_ratio": 0.64,
            },
        ),
        input_bar_count=100,
        filtered_limit_open_count=2,
        elapsed_seconds=0.1,
        data_coverage=pd.DataFrame(
            {
                "timeframe": ["30m"],
                "stock_code": ["000001.SZ"],
                "status": ["ok"],
                "expected_rows": [200],
                "missing_rows": [5],
                "coverage_ratio": [0.975],
                "max_missing_gap_minutes": [60],
                "max_missing_gap_start_at": ["2026-05-25 10:30"],
                "max_missing_gap_end_at": ["2026-05-25 11:30"],
            }
        ),
        strategy_stats=pd.DataFrame(),
        symbol_stats=pd.DataFrame(),
        side_stats=pd.DataFrame(),
        exit_reason_stats=pd.DataFrame(
            {
                "exit_reason": ["take_profit", "stop_loss"],
                "trade_count": [7.0, 5.0],
                "win_rate": [1.0, 0.0],
                "total_return": [0.16, -0.08],
            }
        ),
        monthly_returns=pd.DataFrame(),
        limit_filter_audit=pd.DataFrame({"stock_code": ["000001.SZ"], "status": ["ok"], "filtered_days": [2]}),
    )

    output_dir = save_single_strategy_experiment(result)

    html = (output_dir / "experiment_report.html").read_text(encoding="utf-8")
    assert "<!doctype html>" in html
    assert "single-html-report" in html
    assert "核心绩效" in html
    assert "诊断处理顺序" in html
    assert "复盘路径" in html
    assert "风险画像" in html
    assert "净值与回撤" in html
    assert "净值曲线以 1.0 为基准" in html
    assert 'class="equity-chart"' in html
    assert 'aria-label="净值曲线"' in html
    assert 'aria-label="回撤曲线"' in html
    assert "1.00 基准线" in html
    assert "-4.63%" in html
    assert "数据质量概览" in html
    assert "先确认本次实际 K 线覆盖、缺口和涨跌停开盘过滤影响" in html
    assert "加权覆盖率" in html
    assert "97.50%" in html
    assert "缺失K数" in html
    assert ">5<" in html
    assert "最大连续缺口" in html
    assert "60" in html
    assert "涨跌停过滤" in html
    assert "订单漏斗" in html
    assert "退出结构" in html
    assert "重点证据文件" in html
    assert "产物索引" in html
    assert "总收益" in html
    assert 'class="report-table"' in html
    assert "text-align:center" in html
    assert "<th>平仓原因</th>" in html
    assert "<th>成交数</th>" in html
    assert "8.00%" in html
    assert "36.00%" in html
    assert "未触发（not_triggered）" in html
    assert "末端假突破风险（terminal_false_breakout_risk）" in html
    assert "止盈（take_profit）" in html
    assert "∞" in html
    assert "experiment_diagnostics.csv" in html
    manifest = pd.read_csv(output_dir / "artifact_manifest.csv").set_index("file_name")
    assert manifest.loc["experiment_report.html", "category"] == "阅读入口"
    assert manifest.loc["experiment_report.html", "priority"] == 0
