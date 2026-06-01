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
