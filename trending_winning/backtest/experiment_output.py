from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from html import escape
from math import isfinite
from numbers import Real
from pathlib import Path

import pandas as pd

from trending_winning.backtest.experiment_cases import json_dump, json_ready, sweep_case_config_records, write_jsonl
from trending_winning.backtest.drawdown import drawdown_curve, drawdown_episodes, price_path_drawdown_inputs
from trending_winning.backtest.experiment_diagnostics import (
    case_diagnostic_statistics,
    diagnostic_action_plan,
    experiment_diagnostic_report,
)
from trending_winning.backtest.experiment_models import (
    PortfolioBenchmarkReport,
    PortfolioExperimentConfig,
    PortfolioExperimentResult,
    PortfolioSweepResult,
    SingleStrategyExperimentConfig,
    SingleStrategyExperimentResult,
    SingleStrategySweepResult,
)
from trending_winning.backtest.periods import compute_period_return_statistics
from trending_winning.backtest.reason_labels import exit_reason_label, reason_label_with_code
from trending_winning.backtest.reporting import trade_path_distribution_statistics
from trending_winning.backtest.sweep_analysis import (
    parameter_summary_table as _build_parameter_summary_table,
    pareto_sweep_table as _build_pareto_sweep_table,
)
from trending_winning.backtest.sweep_summary import sweep_summary_statistics as _build_sweep_summary_statistics
from trending_winning.backtest.strategy_space import strategy_space_summary
from trending_winning.data.schema import unique_symbols
from trending_winning.data.summary import summarize_data_management
from trending_winning.data.symbols import DEFAULT_STOCK_NAME_BY_CODE, SYMBOL_METADATA_COLUMNS, load_symbol_metadata


ARTIFACT_MANIFEST_COLUMNS = ["file_name", "category", "priority", "question", "description"]

REPORT_COLUMN_LABELS: dict[str, str] = {
    "acceptance_rate": "订单接受率",
    "avg_risk_adjusted_score": "平均风险质量",
    "best_case_name": "最佳参数组",
    "best_total_return": "最高收益",
    "bucket": "分桶",
    "bucket_order": "排序",
    "case_count": "参数组数",
    "case_name": "参数组",
    "category": "类别",
    "check": "检查项",
    "description": "说明",
    "detail": "详情",
    "diagnostic_primary_issue": "诊断主问题",
    "dimension": "维度",
    "evidence_file": "证据文件",
    "exit_reason": "平仓原因",
    "file_name": "文件名",
    "is_pareto_efficient": "Pareto有效",
    "max_drawdown": "最大回撤",
    "metric": "指标",
    "parameter": "参数",
    "pareto_case_count": "Pareto组数",
    "pareto_hit_rate": "Pareto命中率",
    "pareto_rank": "Pareto排名",
    "positive_return_rate": "正收益率",
    "priority": "优先级",
    "question": "回答的问题",
    "reason": "原因",
    "risk_adjusted_rank": "风险质量排名",
    "risk_adjusted_score": "风险质量分",
    "status": "状态",
    "sweep_rank": "综合排名",
    "threshold": "门槛",
    "total_return": "总收益",
    "trade_count": "成交数",
    "value": "取值",
    "win_rate": "胜率",
    "worst_total_return": "最低收益",
}

REPORT_VALUE_LABELS: dict[str, dict[str, str]] = {
    "side": {"long": "多头", "short": "空头"},
    "status": {"accepted": "接受", "rejected": "拒绝", "passed": "通过", "failed": "失败"},
}


def save_single_strategy_experiment(result: SingleStrategyExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json_dump(json_ready(asdict(result.config))))
    stats = _experiment_stats_payload(result)
    (output_dir / "stats.json").write_text(json_dump(json_ready(stats)))
    diagnostics = _experiment_diagnostic_report(result, stats)
    diagnostics.to_csv(output_dir / "experiment_diagnostics.csv", index=False)
    action_plan = diagnostic_action_plan(diagnostics)
    action_plan.to_csv(output_dir / "diagnostic_action_plan.csv", index=False)
    _write_common_experiment_outputs(output_dir, result)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.signal_lifecycle_stats.to_csv(output_dir / "signal_lifecycle_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    _experiment_trade_path_distribution(result).to_csv(output_dir / "trade_path_distribution.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "setup_strategy_filter_stats.csv", index=False)
    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    manifest = _artifact_manifest("experiment")
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    _write_experiment_report(output_dir, result, stats=stats, diagnostics=diagnostics, action_plan=action_plan, manifest=manifest)
    return output_dir


def save_portfolio_experiment(result: PortfolioExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json_dump(json_ready(asdict(result.config))))
    stats = _experiment_stats_payload(result)
    (output_dir / "stats.json").write_text(json_dump(json_ready(stats)))
    diagnostics = _experiment_diagnostic_report(result, stats)
    diagnostics.to_csv(output_dir / "experiment_diagnostics.csv", index=False)
    action_plan = diagnostic_action_plan(diagnostics)
    action_plan.to_csv(output_dir / "diagnostic_action_plan.csv", index=False)
    _write_common_experiment_outputs(output_dir, result)
    result.strategy_stats.to_csv(output_dir / "strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "symbol_stats.csv", index=False)
    result.side_stats.to_csv(output_dir / "side_stats.csv", index=False)
    result.exit_reason_stats.to_csv(output_dir / "exit_reason_stats.csv", index=False)
    result.signal_lifecycle_stats.to_csv(output_dir / "signal_lifecycle_stats.csv", index=False)
    result.event_type_stats.to_csv(output_dir / "event_type_stats.csv", index=False)
    _experiment_trade_path_distribution(result).to_csv(output_dir / "trade_path_distribution.csv", index=False)
    result.order_decision_stats.to_csv(output_dir / "order_decision_stats.csv", index=False)
    result.strategy_filter_stats.to_csv(output_dir / "strategy_filter_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "setup_strategy_filter_stats.csv", index=False)
    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    manifest = _artifact_manifest("experiment")
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    _write_experiment_report(output_dir, result, stats=stats, diagnostics=diagnostics, action_plan=action_plan, manifest=manifest)
    return output_dir


def save_portfolio_benchmark(config: PortfolioExperimentConfig, report: PortfolioBenchmarkReport) -> Path:
    output_dir = Path(config.output_dir or f"runs/{config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json_dump(json_ready(asdict(report))))
    return output_dir


def save_portfolio_sweep(result: PortfolioSweepResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sweep_outputs(output_dir, result)
    return output_dir


def save_single_strategy_sweep(result: SingleStrategySweepResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sweep_outputs(output_dir, result)
    return output_dir


def symbol_metadata_for_config(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> pd.DataFrame:
    """把实验涉及的股票名称随结果一起保存，避免统计表脱离代码名称映射。"""
    metadata = load_symbol_metadata(config.data_root)
    metadata_by_symbol = {str(row.stock_code): row for row in metadata.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for symbol in unique_symbols(tuple(config.symbols)):
        if symbol in metadata_by_symbol:
            record = metadata_by_symbol[symbol]
            rows.append(
                {
                    "stock_code": symbol,
                    "stock_name": str(record.stock_name),
                    "source": str(record.source),
                    "path": str(record.path),
                }
            )
            continue
        name = DEFAULT_STOCK_NAME_BY_CODE.get(symbol)
        if name:
            rows.append({"stock_code": symbol, "stock_name": name, "source": "default_builtin", "path": ""})
    return pd.DataFrame(rows, columns=pd.Index(SYMBOL_METADATA_COLUMNS))


def _experiment_stats_payload(result: SingleStrategyExperimentResult | PortfolioExperimentResult) -> dict[str, object]:
    stats = dict(result.backtest.stats)
    stats.update(
        summarize_data_management(
            result.data_coverage,
            result.limit_filter_audit,
            filtered_limit_open_count=result.filtered_limit_open_count,
            data_inventory=result.data_inventory,
            min_coverage_ratio=result.config.min_coverage_ratio,
        )
    )
    stats.update(compute_period_return_statistics(result.monthly_returns, prefix="monthly"))
    stats["elapsed_seconds"] = float(result.elapsed_seconds)
    return stats


def _write_common_experiment_outputs(
    output_dir: Path,
    result: SingleStrategyExperimentResult | PortfolioExperimentResult,
) -> None:
    result.backtest.trades.to_csv(output_dir / "trades.csv", index=False)
    result.backtest.order_decisions.to_csv(output_dir / "order_decisions.csv", index=False)
    result.backtest.strategy_filter_decisions.to_csv(output_dir / "strategy_filter_decisions.csv", index=False)
    result.backtest.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    drawdown_curve(result.backtest.equity_curve).to_csv(output_dir / "drawdown_curve.csv", index=False)
    _experiment_drawdown_episodes(result.backtest.equity_curve).to_csv(output_dir / "drawdown_episodes.csv", index=False)
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.data_gap_episodes.to_csv(output_dir / "data_gap_episodes.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    strategy_space_summary(result.config).to_csv(output_dir / "strategy_space.csv", index=False)


def _experiment_drawdown_episodes(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return drawdown_episodes(pd.DataFrame(), pd.Series(dtype=float))
    drawdown_data, drawdown_value = price_path_drawdown_inputs(equity_curve, equity_curve["net_value"])
    return drawdown_episodes(drawdown_data, drawdown_value, limit=20)


def _experiment_trade_path_distribution(result: SingleStrategyExperimentResult | PortfolioExperimentResult) -> pd.DataFrame:
    if not result.trade_path_distribution_stats.empty:
        return result.trade_path_distribution_stats
    return trade_path_distribution_statistics(result.backtest.trades)


def _experiment_diagnostic_report(
    result: SingleStrategyExperimentResult | PortfolioExperimentResult,
    stats: Mapping[str, object],
) -> pd.DataFrame:
    if not result.diagnostic_report.empty:
        return result.diagnostic_report
    return experiment_diagnostic_report(stats, data_coverage=result.data_coverage)


def _write_sweep_outputs(output_dir: Path, result: PortfolioSweepResult | SingleStrategySweepResult) -> None:
    config_payload = json_ready(asdict(result.config))
    config_payload["sweep_grid"] = json_ready(result.grid)
    case_diagnostics = _case_diagnostics(result)
    summary = _sweep_summary_statistics(result, case_diagnostics=case_diagnostics)
    pareto = _pareto_sweep_table(result.table)
    parameter_summary = _parameter_summary_table(result)
    manifest = _artifact_manifest("sweep")
    (output_dir / "config.json").write_text(json_dump(config_payload))
    (output_dir / "summary.json").write_text(json_dump(json_ready(summary)))
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    pareto.to_csv(output_dir / "pareto.csv", index=False)
    parameter_summary.to_csv(output_dir / "parameter_summary.csv", index=False)
    case_diagnostics.to_csv(output_dir / "case_diagnostics.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "case_strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "case_detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "case_setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "case_symbol_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "case_setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "case_setup_strategy_filter_stats.csv", index=False)
    write_jsonl(output_dir / "case_configs.jsonl", sweep_case_config_records(result))
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.data_gap_episodes.to_csv(output_dir / "data_gap_episodes.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)
    manifest.to_csv(output_dir / "artifact_manifest.csv", index=False)
    _write_sweep_report(
        output_dir,
        result,
        summary=summary,
        pareto=pareto,
        parameter_summary=parameter_summary,
        case_diagnostics=case_diagnostics,
        manifest=manifest,
    )


def _artifact_manifest(kind: str) -> pd.DataFrame:
    """保存结果目录的阅读索引，帮助用户先看关键文件再下钻明细。"""
    rows = _sweep_artifact_rows() if kind == "sweep" else _experiment_artifact_rows()
    return pd.DataFrame(rows, columns=pd.Index(ARTIFACT_MANIFEST_COLUMNS))


def _experiment_artifact_rows() -> list[tuple[str, str, int, str, str]]:
    return [
        (
            "experiment_report.html",
            "阅读入口",
            0,
            "能否先用一个页面看懂本次回测？",
            "静态 HTML 总览，汇总核心绩效、诊断处理顺序和产物索引。",
        ),
        (
            "artifact_manifest.csv",
            "阅读入口",
            1,
            "这个目录里的文件先看什么？",
            "当前结果目录的索引，说明每个文件回答的问题。",
        ),
        (
            "strategy_space.csv",
            "运行前复核",
            1,
            "本次到底启用了哪些策略边界？",
            "列出样本、形态、触发、过滤、退出、仓位、统计和失效空间。",
        ),
        ("config.json", "复现实验", 1, "本次参数如何复现？", "完整保存本次实验配置。"),
        ("stats.json", "核心统计", 1, "收益、回撤、成交和拒单概况是什么？", "实验级聚合统计。"),
        (
            "experiment_diagnostics.csv",
            "核心统计",
            1,
            "优先检查数据、信号、风控还是仓位？",
            "把回测质量问题汇总成可复核诊断。",
        ),
        (
            "diagnostic_action_plan.csv",
            "核心统计",
            1,
            "诊断问题应该按什么顺序处理？",
            "按失败和关注项排序，给出处理动作和对应证据文件。",
        ),
        ("trades.csv", "成交明细", 1, "每笔交易怎样开仓和平仓？", "逐笔成交、退出原因和盈亏明细。"),
        ("equity_curve.csv", "净值与回撤", 1, "组合资产净值如何逐 K 线变化？", "净值曲线原始序列。"),
        (
            "drawdown_curve.csv",
            "净值与回撤",
            1,
            "组合资产价格波动产生了多大回撤？",
            "按净值路径计算的连续回撤曲线。",
        ),
        (
            "drawdown_episodes.csv",
            "净值与回撤",
            1,
            "主要回撤区间从哪里开始、在哪里见底？",
            "按组合资产净值路径拆分的重点回撤区间。",
        ),
        (
            "order_decisions.csv",
            "订单与过滤",
            1,
            "信号触发后为什么成交或没有成交？",
            "撮合层订单接受、拒绝、风险和追价原因。",
        ),
        (
            "strategy_filter_decisions.csv",
            "订单与过滤",
            1,
            "信号为什么被策略过滤？",
            "策略层过滤原因，例如末端假突破风险。",
        ),
        (
            "data_inventory.csv",
            "数据质量",
            1,
            "本次实际读到了哪些 K 线缓存？",
            "本地数据文件、行数、时间范围和快照签名。",
        ),
        (
            "data_coverage.csv",
            "数据质量",
            1,
            "样本覆盖是否足够做回测？",
            "每个标的周期的覆盖率和缺口数量。",
        ),
        (
            "data_gap_episodes.csv",
            "数据质量",
            1,
            "哪段 K 线连续缺失？",
            "逐段列出每段连续缺失 K 线的起止时间、根数和文件路径。",
        ),
        ("strategy_stats.csv", "分组统计", 2, "不同策略贡献如何？", "按策略信号统计交易表现。"),
        ("detector_stats.csv", "分组统计", 2, "不同识别模块贡献如何？", "按趋势、通道等识别模块统计表现。"),
        ("setup_stats.csv", "分组统计", 2, "不同 setup 的质量如何？", "按 setup 名称统计交易表现。"),
        ("symbol_stats.csv", "分组统计", 2, "不同股票贡献如何？", "按股票统计交易表现。"),
        ("side_stats.csv", "分组统计", 2, "多头和空头表现差异如何？", "按交易方向统计表现。"),
        ("exit_reason_stats.csv", "分组统计", 2, "主要靠什么方式平仓？", "按止损、止盈、持仓到期等退出原因统计。"),
        ("signal_lifecycle_stats.csv", "分组统计", 2, "信号从出现到退出的链路是否顺畅？", "按信号生命周期统计。"),
        ("event_type_stats.csv", "分组统计", 2, "事件类型分布是否异常？", "按事件类型统计交易表现。"),
        ("trade_path_distribution.csv", "分组统计", 2, "交易路径中的波动和回撤长什么样？", "逐笔交易路径分布统计。"),
        ("monthly_returns.csv", "分组统计", 2, "月度收益是否稳定？", "按月份统计收益。"),
        (
            "order_decision_stats.csv",
            "订单与过滤",
            2,
            "撮合层拒单集中在哪些原因？",
            "订单接受率、拒绝率和风险指标汇总。",
        ),
        (
            "strategy_filter_stats.csv",
            "订单与过滤",
            2,
            "策略过滤集中在哪些原因？",
            "策略层过滤接受率和拒绝原因汇总。",
        ),
        (
            "setup_order_decision_stats.csv",
            "订单与过滤",
            2,
            "哪个 setup 更容易被撮合层拒绝？",
            "按 setup 拆分订单决策统计。",
        ),
        (
            "setup_strategy_filter_stats.csv",
            "订单与过滤",
            2,
            "哪个 setup 更容易被策略过滤？",
            "按 setup 拆分策略过滤统计。",
        ),
        (
            "limit_filter_audit.csv",
            "数据质量",
            2,
            "涨跌停开盘过滤影响了哪些样本？",
            "记录涨跌停开盘样本过滤过程。",
        ),
        ("symbol_metadata.csv", "标的信息", 2, "股票代码对应什么名称？", "股票名称和来源路径。"),
    ]


def _write_experiment_report(
    output_dir: Path,
    result: SingleStrategyExperimentResult | PortfolioExperimentResult,
    *,
    stats: Mapping[str, object],
    diagnostics: pd.DataFrame,
    action_plan: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    html = _experiment_report_html(
        result,
        stats=stats,
        diagnostics=diagnostics,
        action_plan=action_plan,
        manifest=manifest,
    )
    (output_dir / "experiment_report.html").write_text(html, encoding="utf-8")


def _experiment_report_html(
    result: SingleStrategyExperimentResult | PortfolioExperimentResult,
    *,
    stats: Mapping[str, object],
    diagnostics: pd.DataFrame,
    action_plan: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    title = str(result.config.name)
    metric_cards = "".join(
        _metric_card(label, stats.get(key), key)
        for label, key in (
            ("交易数", "trade_count"),
            ("总收益", "total_return"),
            ("最大回撤", "max_drawdown"),
            ("胜率", "win_rate"),
            ("盈亏因子", "profit_factor"),
            ("订单接受率", "acceptance_rate"),
        )
    )
    risk_cards = "".join(
        _metric_card(label, stats.get(key), key, note)
        for label, key, note in (
            ("当前回撤", "current_drawdown", "最新净值相对历史高点的回撤。"),
            ("回撤压力", "ulcer_index", "净值水下时间和深度的综合压力。"),
            ("场内时间", "exposure_bar_ratio", "持仓 K 数占样本 K 数的比例。"),
            ("平均现金", "avg_cash_ratio", "组合层平均未使用现金比例。"),
        )
    )
    order_cards = "".join(
        _metric_card(label, stats.get(key), key, note)
        for label, key, note in (
            ("订单总数", "order_count", "进入撮合层的候选订单。"),
            ("成交交易", "trade_count", "真实进入持仓并完成退出的交易。"),
            ("订单接受率", "acceptance_rate", "撮合和风控后真正开仓的比例。"),
            ("策略过滤率", "strategy_filter_rejection_rate", "策略层提前拒绝开仓的比例。"),
        )
    )
    equity_drawdown_panel = _equity_drawdown_panel(result.backtest.equity_curve)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} 回测总览</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#fff; --ink:#102033; --muted:#5f6f84; --line:#dce4ee; --blue:#1769aa; --red:#b42318; --green:#0f7a55; --radius:8px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.58; }}
    .wrap {{ width:min(1180px, calc(100vw - 40px)); margin:0 auto; }}
    header {{ padding:34px 0 24px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:34px; line-height:1.18; letter-spacing:0; }}
    .lead {{ margin:0; color:var(--muted); }}
    .status-strip {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
    .status-badge {{ padding:6px 10px; border:1px solid var(--line); border-radius:999px; background:#f8fafc; color:#24364d; font-size:13px; font-weight:700; }}
    .status-fail {{ color:var(--red); background:#fff7f6; border-color:#f2c8c3; }}
    .status-watch {{ color:#a15c00; background:#fff8eb; border-color:#efd2a5; }}
    .status-pass {{ color:var(--green); background:#f0faf5; border-color:#b9e5cc; }}
    main {{ padding:22px 0 52px; }}
    section {{ margin-top:18px; padding:22px; border:1px solid var(--line); border-radius:var(--radius); background:var(--panel); }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    .section-note {{ margin:0 0 14px; color:var(--muted); font-size:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; }}
    .metric {{ padding:14px; border:1px solid var(--line); border-radius:var(--radius); background:#fbfcfe; }}
    .metric b {{ display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }}
    .metric span {{ font-size:22px; font-weight:760; }}
    .metric small {{ display:block; margin-top:6px; color:var(--muted); font-size:12px; }}
    .chart-grid {{ display:grid; grid-template-columns:minmax(0,1.35fr) minmax(0,1fr); gap:14px; }}
    .chart-card {{ padding:14px; border:1px solid var(--line); border-radius:var(--radius); background:#fbfcfe; }}
    .chart-card strong {{ display:block; margin-bottom:4px; color:#24364d; }}
    .chart-note {{ margin:0 0 10px; color:var(--muted); font-size:13px; }}
    .equity-chart {{ display:block; width:100%; height:auto; }}
    .chart-axis {{ stroke:#d8e1eb; stroke-width:1; }}
    .chart-baseline {{ stroke:#7b8794; stroke-width:1.2; stroke-dasharray:5 5; }}
    .chart-equity-line {{ fill:none; stroke:#1769aa; stroke-width:3; stroke-linecap:round; stroke-linejoin:round; }}
    .chart-drawdown-line {{ fill:none; stroke:#b42318; stroke-width:3; stroke-linecap:round; stroke-linejoin:round; }}
    .chart-drawdown-area {{ fill:#fde8e4; opacity:.85; }}
    .chart-label {{ fill:#5f6f84; font-size:12px; }}
    .chart-value {{ fill:#24364d; font-size:13px; font-weight:700; }}
    .review-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .review-card {{ padding:14px; border:1px solid var(--line); border-radius:var(--radius); background:#fbfcfe; }}
    .review-card strong {{ display:block; margin-bottom:6px; color:#24364d; }}
    .review-card p {{ margin:6px 0; color:var(--muted); font-size:13px; }}
    .review-card em {{ display:inline-block; margin-bottom:6px; color:var(--blue); font-style:normal; font-weight:700; font-size:12px; }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }}
    .evidence-list {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .evidence-list a {{ padding:7px 10px; border:1px solid #cfe0f2; border-radius:999px; background:#f7fbff; font-size:13px; }}
    .report-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .report-table th,.report-table td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:center; vertical-align:middle; }}
    .report-table td.num {{ font-variant-numeric:tabular-nums; }}
    th {{ background:#f8fafc; color:#24364d; }}
    a {{ color:var(--blue); text-decoration:none; }}
    .empty {{ color:var(--muted); }}
    @media (max-width: 920px) {{ .metrics,.review-grid,.two-col,.chart-grid {{ grid-template-columns:1fr; }} section {{ padding:16px; }} .wrap {{ width:min(100vw - 24px, 1180px); }} }}
  </style>
</head>
<body>
<header><div class="wrap"><h1>{escape(title)} 回测总览</h1><p class="lead">{escape(_experiment_scope_text(result))}</p>{_diagnostic_status_badges(diagnostics)}</div></header>
<main class="wrap">
  <section><h2>核心绩效</h2><div class="metrics">{metric_cards}</div></section>
  <section><h2>净值与回撤</h2><p class="section-note">净值曲线以 1.0 为基准，回撤按组合资产净值路径计算。</p>{equity_drawdown_panel}</section>
  <section><h2>复盘路径</h2><p class="section-note">按诊断严重程度排序，先处理会改变结论的问题，再下钻对应证据文件。</p>{_review_path_cards(action_plan)}</section>
  <section><h2>风险画像</h2><div class="metrics">{risk_cards}</div></section>
  <section><h2>订单漏斗</h2><div class="metrics">{order_cards}</div><div class="two-col">{_reason_panel("撮合主因", stats, "primary_rejected_reason", "primary_rejected_reason_count", "primary_rejected_reason_rate")}{_reason_panel("策略过滤主因", stats, "primary_strategy_rejected_reason", "primary_strategy_rejected_reason_count", "primary_strategy_rejected_reason_rate")}</div></section>
  <section><h2>退出结构</h2>{_compact_report_table(result.exit_reason_stats, ["exit_reason", "trade_count", "win_rate", "total_return"])}</section>
  <section><h2>重点证据文件</h2>{_evidence_file_panel(action_plan)}</section>
  <section><h2>诊断处理顺序</h2>{_html_table(action_plan, link_files=True)}</section>
  <section><h2>实验诊断摘要</h2>{_html_table(diagnostics)}</section>
  <section><h2>产物索引</h2>{_html_table(manifest, link_files=True)}</section>
</main>
</body>
</html>
"""


def _equity_drawdown_panel(equity_curve: pd.DataFrame) -> str:
    if equity_curve.empty or "net_value" not in equity_curve.columns:
        return '<p class="empty">暂无净值曲线。</p>'
    values = pd.to_numeric(equity_curve["net_value"], errors="coerce").dropna().reset_index(drop=True)
    if values.empty:
        return '<p class="empty">暂无有效净值点。</p>'

    running_peak = values.cummax()
    drawdown = values / running_peak - 1.0
    equity_svg = _line_chart_svg(
        values,
        aria_label="净值曲线",
        css_class="chart-equity-line",
        baseline=1.0,
        baseline_label="1.00 基准线",
    )
    drawdown_svg = _line_chart_svg(
        drawdown,
        aria_label="回撤曲线",
        css_class="chart-drawdown-line",
        baseline=0.0,
        baseline_label="0.00%",
        fill_to_baseline=True,
        value_key="drawdown",
    )
    latest_equity = _format_report_value("net_value", values.iloc[-1])
    latest_drawdown = _format_report_value("current_drawdown", drawdown.iloc[-1])
    max_drawdown = _format_report_value("max_drawdown", drawdown.min())
    return (
        '<div class="chart-grid">'
        '<div class="chart-card">'
        "<strong>净值曲线</strong>"
        f'<p class="chart-note">期末净值 {escape(latest_equity)}，1.00 基准线用于确认收益尺度。</p>'
        f"{equity_svg}"
        "</div>"
        '<div class="chart-card">'
        "<strong>回撤曲线</strong>"
        f'<p class="chart-note">当前回撤 {escape(latest_drawdown)}，最大回撤 {escape(max_drawdown)}。</p>'
        f"{drawdown_svg}"
        "</div>"
        "</div>"
    )


def _line_chart_svg(
    values: pd.Series,
    *,
    aria_label: str,
    css_class: str,
    baseline: float,
    baseline_label: str,
    fill_to_baseline: bool = False,
    value_key: str = "net_value",
) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if clean.empty:
        return '<p class="empty">暂无有效曲线。</p>'
    width = 720.0
    height = 220.0
    pad_left = 46.0
    pad_right = 18.0
    pad_top = 18.0
    pad_bottom = 34.0
    y_min = float(min(clean.min(), baseline))
    y_max = float(max(clean.max(), baseline))
    if abs(y_max - y_min) < 1e-12:
        y_min -= 0.01
        y_max += 0.01

    def x_at(pos: int) -> float:
        if len(clean) == 1:
            return pad_left
        return pad_left + (width - pad_left - pad_right) * pos / (len(clean) - 1)

    def y_at(value: float) -> float:
        return pad_top + (y_max - value) / (y_max - y_min) * (height - pad_top - pad_bottom)

    points = [(x_at(pos), y_at(float(value))) for pos, value in enumerate(clean)]
    point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    baseline_y = y_at(float(baseline))
    y_max_label = _format_report_value(value_key, y_max)
    y_min_label = _format_report_value(value_key, y_min)
    fill_path = ""
    if fill_to_baseline and len(points) >= 2:
        first_x = points[0][0]
        last_x = points[-1][0]
        fill_path = (
            f'<polygon class="chart-drawdown-area" points="{first_x:.2f},{baseline_y:.2f} '
            f'{point_text} {last_x:.2f},{baseline_y:.2f}" />'
        )
    return (
        f'<svg class="equity-chart" viewBox="0 0 {int(width)} {int(height)}" role="img" '
        f'aria-label="{escape(aria_label, quote=True)}">'
        f'<line class="chart-axis" x1="{pad_left:.2f}" y1="{pad_top:.2f}" x2="{pad_left:.2f}" y2="{height - pad_bottom:.2f}" />'
        f'<line class="chart-axis" x1="{pad_left:.2f}" y1="{height - pad_bottom:.2f}" x2="{width - pad_right:.2f}" y2="{height - pad_bottom:.2f}" />'
        f'<line class="chart-baseline" x1="{pad_left:.2f}" y1="{baseline_y:.2f}" x2="{width - pad_right:.2f}" y2="{baseline_y:.2f}" />'
        f"{fill_path}"
        f'<polyline class="{escape(css_class, quote=True)}" points="{point_text}" />'
        f'<text class="chart-label" x="8" y="{pad_top + 4:.2f}">{escape(y_max_label)}</text>'
        f'<text class="chart-label" x="8" y="{height - pad_bottom + 4:.2f}">{escape(y_min_label)}</text>'
        f'<text class="chart-value" x="{pad_left + 4:.2f}" y="{baseline_y - 6:.2f}">{escape(baseline_label)}</text>'
        "</svg>"
    )


def _experiment_scope_text(result: SingleStrategyExperimentResult | PortfolioExperimentResult) -> str:
    config = result.config
    symbols = "、".join(str(symbol) for symbol in config.symbols[:5])
    if len(config.symbols) > 5:
        symbols += f" 等 {len(config.symbols)} 个标的"
    return f"{config.timeframe} | {config.start} 至 {config.end} | {symbols}"


def _metric_card(label: str, value: object, key: str, note: str = "") -> str:
    note_html = f"<small>{escape(note)}</small>" if note else ""
    return (
        f'<div class="metric"><b>{escape(label)}</b><span>{escape(_format_report_value(key, value))}</span>'
        f"{note_html}</div>"
    )


def _diagnostic_status_badges(diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty or "status" not in diagnostics.columns:
        return ""
    status = diagnostics["status"].fillna("").astype(str)
    failed = int(status.eq("失败").sum())
    attention = int(status.eq("关注").sum())
    passed = int(status.eq("通过").sum())
    return (
        '<div class="status-strip">'
        f'<span class="status-badge status-fail">失败 {failed}</span>'
        f'<span class="status-badge status-watch">关注 {attention}</span>'
        f'<span class="status-badge status-pass">通过 {passed}</span>'
        "</div>"
    )


def _review_path_cards(action_plan: pd.DataFrame) -> str:
    if action_plan.empty:
        return '<p class="empty">暂无优先问题，先查看核心绩效和产物索引。</p>'
    cards: list[str] = []
    for row in action_plan.head(3).to_dict("records"):
        priority = escape(str(row.get("priority", "")))
        status = escape(str(row.get("status", "")))
        check = escape(str(row.get("check", "")))
        action = escape(str(row.get("action", "")))
        detail = escape(str(row.get("detail", "")))
        evidence = _evidence_links(row.get("evidence_file", ""))
        cards.append(
            '<div class="review-card">'
            f"<em>优先 {priority} · {status}</em>"
            f"<strong>{check}</strong>"
            f"<p>{action}</p>"
            f"<p>{detail}</p>"
            f"<p>{evidence}</p>"
            "</div>"
        )
    return f'<div class="review-grid">{"".join(cards)}</div>'


def _reason_panel(
    title: str,
    stats: Mapping[str, object],
    reason_key: str,
    count_key: str,
    rate_key: str,
) -> str:
    reason = reason_label_with_code(stats.get(reason_key)) or "暂无"
    count = _format_report_value(count_key, stats.get(count_key))
    rate = _format_report_value(rate_key, stats.get(rate_key))
    return (
        '<div class="review-card">'
        f"<strong>{escape(title)}</strong>"
        f"<p>{escape(reason)}</p>"
        f"<p>数量 {escape(count)} · 占比 {escape(rate)}</p>"
        "</div>"
    )


def _compact_report_table(frame: pd.DataFrame, preferred_columns: list[str], *, max_rows: int = 6) -> str:
    if frame.empty:
        return '<p class="empty">暂无分组统计。</p>'
    columns = [column for column in preferred_columns if column in frame.columns]
    if not columns:
        columns = list(frame.columns[:4])
    return _html_table(frame.loc[:, columns].head(max_rows))


def _evidence_file_panel(action_plan: pd.DataFrame) -> str:
    files: list[str] = []
    if not action_plan.empty and "evidence_file" in action_plan.columns:
        for value in action_plan["evidence_file"].head(5):
            files.extend(item.strip() for item in str(value).split(";") if item.strip())
    if not files:
        files = ["experiment_diagnostics.csv", "artifact_manifest.csv", "stats.json"]
    unique_files = list(dict.fromkeys(files))
    links = "".join(_file_link(file_name) for file_name in unique_files[:8])
    return f'<div class="evidence-list">{links}</div>'


def _html_table(frame: pd.DataFrame, *, link_files: bool = False) -> str:
    if frame.empty:
        return '<p class="empty">暂无记录。</p>'
    data = frame.copy()
    if link_files and "file_name" in data.columns:
        data["file_name"] = data["file_name"].map(_file_link)
    if link_files and "evidence_file" in data.columns:
        data["evidence_file"] = data["evidence_file"].map(_evidence_links)
    headers = "".join(f"<th>{escape(_report_column_label(str(column)))}</th>" for column in data.columns)
    rows = []
    for record in data.to_dict("records"):
        cells = "".join(
            f'<td class="{_html_cell_class(value)}">{_html_cell(column, value)}</td>' for column, value in record.items()
        )
        rows.append(f"<tr>{cells}</tr>")
    return f'<table class="report-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _file_link(value: object) -> str:
    text = str(value)
    return f'<a href="{escape(text, quote=True)}">{escape(text)}</a>'


def _evidence_links(value: object) -> str:
    files = [item.strip() for item in str(value).split(";") if item.strip()]
    return "; ".join(_file_link(file_name) for file_name in files)


def _html_cell(column: str, value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.startswith("<a "):
        return text
    if isinstance(value, bool):
        return "是" if value else "否"
    normalized_column = str(column)
    if normalized_column == "exit_reason":
        return escape(_label_with_code(exit_reason_label(value), value))
    if normalized_column in {"reason", "primary_rejected_reason", "primary_strategy_rejected_reason"}:
        return escape(reason_label_with_code(value))
    labeled_value = REPORT_VALUE_LABELS.get(normalized_column, {}).get(text)
    if labeled_value:
        return escape(labeled_value)
    if isinstance(value, Real):
        return escape(_format_report_value(str(column), value))
    return escape(text)


def _report_column_label(column: str) -> str:
    return REPORT_COLUMN_LABELS.get(column, column)


def _html_cell_class(value: object) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        return "num"
    return "text"


def _label_with_code(label: str, value: object) -> str:
    code = str(value or "").strip()
    if not code:
        return ""
    if not label or label == code:
        return code
    return f"{label}（{code}）"


def _format_report_value(key: str, value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not isfinite(numeric):
        return "∞" if numeric > 0 else "-∞"
    if (
        key.endswith("_rate")
        or key.endswith("_ratio")
        or key.endswith("_pct")
        or key.endswith("_return")
        or key.endswith("_drawdown")
        or key in {"win_rate", "max_drawdown"}
    ):
        return f"{numeric:.2%}"
    if abs(numeric - round(numeric)) < 1e-12 and key.endswith("_count"):
        return f"{int(round(numeric)):,}"
    if abs(numeric - round(numeric)) < 1e-12 and (
        key.endswith("_rank") or key in {"priority", "sweep_rank", "pareto_rank"}
    ):
        return f"{int(round(numeric)):,}"
    return f"{numeric:.2f}"


def _sweep_artifact_rows() -> list[tuple[str, str, int, str, str]]:
    return [
        (
            "sweep_report.html",
            "阅读入口",
            0,
            "能否先用一个页面看懂参数遍历？",
            "静态 HTML 总览，汇总最优参数组、风险质量候选、Pareto 候选、参数影响和诊断概览。",
        ),
        (
            "artifact_manifest.csv",
            "阅读入口",
            1,
            "这个目录里的文件先看什么？",
            "当前参数遍历目录的索引，说明每个文件回答的问题。",
        ),
        ("config.json", "复现实验", 1, "本次参数遍历如何复现？", "基础配置和 sweep_grid 参数空间。"),
        ("summary.json", "参数遍历", 1, "参数遍历总体质量如何？", "参数组合数量、耗时、数据和信号质量摘要。"),
        (
            "sweep.csv",
            "参数遍历",
            1,
            "先筛选参数组，看收益、回撤、诊断状态和风险质量排名。",
            "每组参数的核心收益、回撤、成交、拒单和排名字段。",
        ),
        ("pareto.csv", "参数遍历", 1, "哪些参数组在收益和风险之间更占优？", "帕累托有效参数组。"),
        (
            "parameter_summary.csv",
            "参数遍历",
            1,
            "单个参数值整体倾向好还是坏？",
            "按参数取值聚合表现、稳定性和拒单指标。",
        ),
        ("case_diagnostics.csv", "参数遍历", 1, "哪些参数组质量不合格？", "每组参数的诊断状态。"),
        (
            "data_inventory.csv",
            "数据质量",
            1,
            "本次实际读到了哪些 K 线缓存？",
            "本地数据文件、行数、时间范围和快照签名。",
        ),
        (
            "data_coverage.csv",
            "数据质量",
            1,
            "样本覆盖是否足够做参数遍历？",
            "每个标的周期的覆盖率和缺口数量。",
        ),
        (
            "data_gap_episodes.csv",
            "数据质量",
            1,
            "哪段 K 线连续缺失？",
            "逐段列出每段连续缺失 K 线的起止时间、根数和文件路径。",
        ),
        (
            "case_configs.jsonl",
            "复现实验",
            2,
            "每个参数组的完整配置是什么？",
            "逐行保存每个 case 的复现配置。",
        ),
        ("case_strategy_stats.csv", "分组统计", 2, "不同参数下策略贡献如何？", "按 case 和策略信号统计表现。"),
        ("case_detector_stats.csv", "分组统计", 2, "不同参数下识别模块贡献如何？", "按 case 和识别模块统计表现。"),
        ("case_setup_stats.csv", "分组统计", 2, "不同参数下 setup 质量如何？", "按 case 和 setup 统计表现。"),
        ("case_symbol_stats.csv", "分组统计", 2, "不同参数下股票贡献如何？", "按 case 和股票统计表现。"),
        (
            "case_setup_order_decision_stats.csv",
            "订单与过滤",
            2,
            "参数变化后哪个 setup 更容易被撮合层拒绝？",
            "按 case 和 setup 拆分订单决策统计。",
        ),
        (
            "case_setup_strategy_filter_stats.csv",
            "订单与过滤",
            2,
            "参数变化后哪个 setup 更容易被策略过滤？",
            "按 case 和 setup 拆分策略过滤统计。",
        ),
        (
            "limit_filter_audit.csv",
            "数据质量",
            2,
            "涨跌停开盘过滤影响了哪些样本？",
            "记录涨跌停开盘样本过滤过程。",
        ),
        ("symbol_metadata.csv", "标的信息", 2, "股票代码对应什么名称？", "股票名称和来源路径。"),
    ]


def _write_sweep_report(
    output_dir: Path,
    result: PortfolioSweepResult | SingleStrategySweepResult,
    *,
    summary: Mapping[str, object],
    pareto: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    case_diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    html = _sweep_report_html(
        result,
        summary=summary,
        pareto=pareto,
        parameter_summary=parameter_summary,
        case_diagnostics=case_diagnostics,
        manifest=manifest,
    )
    (output_dir / "sweep_report.html").write_text(html, encoding="utf-8")


def _sweep_report_html(
    result: PortfolioSweepResult | SingleStrategySweepResult,
    *,
    summary: Mapping[str, object],
    pareto: pd.DataFrame,
    parameter_summary: pd.DataFrame,
    case_diagnostics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    title = str(result.config.name)
    metric_cards = "".join(
        _metric_card(label, summary.get(key), key, note)
        for label, key, note in (
            ("实际运行组", "case_count", "去重后真实执行的参数组数量。"),
            ("原始组合数", "grid_case_count", "用户输入 grid 展开后的组合数。"),
            ("Pareto候选", "pareto_case_count", "收益和风险非支配的第一层候选。"),
            ("最高风险质量", "best_risk_adjusted_score", "风险质量评分越高，综合稳健性越好。"),
            ("耗时", "elapsed_seconds", "本次参数遍历运行耗时，单位秒。"),
            ("订单缓存命中", "order_cache_hit_rate", "复用订单流减少重复计算的比例。"),
        )
    )
    top_cases = _compact_report_table(
        result.table,
        [
            "sweep_rank",
            "risk_adjusted_rank",
            "pareto_rank",
            "case_name",
            "total_return",
            "max_drawdown",
            "trade_count",
            "win_rate",
            "acceptance_rate",
            "diagnostic_primary_issue",
        ],
        max_rows=8,
    )
    risk_cases = _compact_report_table(
        _sort_frame(result.table, ["risk_adjusted_rank", "sweep_rank"]),
        [
            "risk_adjusted_rank",
            "risk_adjusted_score",
            "sweep_rank",
            "case_name",
            "total_return",
            "max_drawdown",
            "trade_count",
            "diagnostic_primary_issue",
        ],
        max_rows=8,
    )
    pareto_cases = _compact_report_table(
        pareto,
        [
            "pareto_rank",
            "sweep_rank",
            "risk_adjusted_rank",
            "case_name",
            "total_return",
            "max_drawdown",
            "trade_count",
            "risk_adjusted_score",
        ],
        max_rows=8,
    )
    parameter_impact = _compact_report_table(
        parameter_summary,
        [
            "parameter",
            "value",
            "case_count",
            "pareto_case_count",
            "pareto_hit_rate",
            "positive_return_rate",
            "best_total_return",
            "worst_total_return",
            "avg_risk_adjusted_score",
            "best_case_name",
        ],
        max_rows=12,
    )
    diagnostics = _compact_report_table(
        _sort_frame(case_diagnostics, ["status", "sweep_rank"]),
        ["sweep_rank", "case_name", "check", "status", "metric", "value", "threshold", "detail"],
        max_rows=12,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)} 参数遍历总览</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#fff; --ink:#102033; --muted:#5f6f84; --line:#dce4ee; --blue:#1769aa; --radius:8px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; line-height:1.58; }}
    .wrap {{ width:min(1180px, calc(100vw - 40px)); margin:0 auto; }}
    header {{ padding:34px 0 24px; background:#fff; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 8px; font-size:34px; line-height:1.18; letter-spacing:0; }}
    .lead {{ margin:0; color:var(--muted); }}
    main {{ padding:22px 0 52px; }}
    section {{ margin-top:18px; padding:22px; border:1px solid var(--line); border-radius:var(--radius); background:var(--panel); }}
    h2 {{ margin:0 0 12px; font-size:22px; }}
    .section-note {{ margin:0 0 14px; color:var(--muted); font-size:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:12px; }}
    .metric {{ padding:14px; border:1px solid var(--line); border-radius:var(--radius); background:#fbfcfe; }}
    .metric b {{ display:block; color:var(--muted); font-size:13px; margin-bottom:6px; }}
    .metric span {{ font-size:22px; font-weight:760; }}
    .metric small {{ display:block; margin-top:6px; color:var(--muted); font-size:12px; }}
    .two-col {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }}
    .evidence-list {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .evidence-list a {{ padding:7px 10px; border:1px solid #cfe0f2; border-radius:999px; background:#f7fbff; font-size:13px; color:var(--blue); text-decoration:none; }}
    .report-table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    .report-table th,.report-table td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:center; vertical-align:middle; }}
    .report-table td.num {{ font-variant-numeric:tabular-nums; }}
    th {{ background:#f8fafc; color:#24364d; }}
    a {{ color:var(--blue); text-decoration:none; }}
    .empty {{ color:var(--muted); }}
    @media (max-width: 920px) {{ .metrics,.two-col {{ grid-template-columns:1fr; }} section {{ padding:16px; }} .wrap {{ width:min(100vw - 24px, 1180px); }} }}
  </style>
</head>
<body>
<header><div class="wrap"><h1>{escape(title)} 参数遍历总览</h1><p class="lead">{escape(_sweep_scope_text(result))}</p></div></header>
<main class="wrap">
  <section><h2>参数遍历总览</h2><div class="metrics">{metric_cards}</div></section>
  <section><h2>最优参数组</h2><p class="section-note">按 sweep_rank 排序，先看收益、回撤、交易数和诊断主问题。</p>{top_cases}</section>
  <section><h2>风险质量候选</h2><p class="section-note">按 risk_adjusted_rank 排序，优先找收益、回撤、样本量更均衡的参数组。</p>{risk_cases}</section>
  <section><h2>Pareto候选</h2>{pareto_cases}</section>
  <section><h2>参数影响</h2>{parameter_impact}</section>
  <section><h2>诊断概览</h2>{diagnostics}</section>
  <section><h2>重点产物</h2>{_sweep_evidence_file_panel()}</section>
  <section><h2>产物索引</h2>{_html_table(manifest, link_files=True)}</section>
</main>
</body>
</html>
"""


def _sweep_scope_text(result: PortfolioSweepResult | SingleStrategySweepResult) -> str:
    config = result.config
    fields = "、".join(str(field) for field in result.grid)
    if not fields:
        fields = "无参数字段"
    return f"{config.timeframe} | {config.start} 至 {config.end} | 参数：{fields}"


def _sweep_evidence_file_panel() -> str:
    files = [
        "sweep.csv",
        "pareto.csv",
        "parameter_summary.csv",
        "case_diagnostics.csv",
        "case_configs.jsonl",
        "artifact_manifest.csv",
    ]
    return f'<div class="evidence-list">{"".join(_file_link(file_name) for file_name in files)}</div>'


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame
    return frame.sort_values(available, ascending=True, kind="mergesort").reset_index(drop=True)


def _sweep_summary_statistics(
    result: PortfolioSweepResult | SingleStrategySweepResult,
    *,
    case_diagnostics: pd.DataFrame | None = None,
) -> dict[str, object]:
    return _build_sweep_summary_statistics(
        table=result.table,
        grid=result.grid,
        elapsed_seconds=result.elapsed_seconds,
        input_bar_count=result.input_bar_count,
        filtered_limit_open_count=result.filtered_limit_open_count,
        strategy_stats=result.strategy_stats,
        detector_stats=result.detector_stats,
        setup_stats=result.setup_stats,
        symbol_stats=result.symbol_stats,
        setup_order_decision_stats=result.setup_order_decision_stats,
        setup_strategy_filter_stats=result.setup_strategy_filter_stats,
        case_diagnostics=case_diagnostics if case_diagnostics is not None else _case_diagnostics(result),
    )


def _pareto_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    return _build_pareto_sweep_table(table)


def _parameter_summary_table(result: PortfolioSweepResult | SingleStrategySweepResult) -> pd.DataFrame:
    return _build_parameter_summary_table(result.table, result.grid)


def _case_diagnostics(result: PortfolioSweepResult | SingleStrategySweepResult) -> pd.DataFrame:
    if not result.case_diagnostics.empty:
        return result.case_diagnostics
    return case_diagnostic_statistics(result.table)
