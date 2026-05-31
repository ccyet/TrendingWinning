from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trending_winning.backtest.experiment_cases import json_dump, json_ready, sweep_case_config_records, write_jsonl
from trending_winning.backtest.drawdown import drawdown_episodes, price_path_drawdown_inputs
from trending_winning.backtest.experiment_diagnostics import (
    case_diagnostic_statistics,
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
from trending_winning.backtest.reporting import trade_path_distribution_statistics
from trending_winning.backtest.sweep_analysis import (
    parameter_summary_table as _build_parameter_summary_table,
    pareto_sweep_table as _build_pareto_sweep_table,
)
from trending_winning.backtest.sweep_summary import sweep_summary_statistics as _build_sweep_summary_statistics
from trending_winning.data.schema import unique_symbols
from trending_winning.data.summary import summarize_data_management
from trending_winning.data.symbols import DEFAULT_STOCK_NAME_BY_CODE, SYMBOL_METADATA_COLUMNS, load_symbol_metadata


def save_single_strategy_experiment(result: SingleStrategyExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json_dump(json_ready(asdict(result.config))))
    stats = _experiment_stats_payload(result)
    (output_dir / "stats.json").write_text(json_dump(json_ready(stats)))
    _experiment_diagnostic_report(result, stats).to_csv(output_dir / "experiment_diagnostics.csv", index=False)
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
    return output_dir


def save_portfolio_experiment(result: PortfolioExperimentResult) -> Path:
    output_dir = Path(result.config.output_dir or f"runs/{result.config.name}").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json_dump(json_ready(asdict(result.config))))
    stats = _experiment_stats_payload(result)
    (output_dir / "stats.json").write_text(json_dump(json_ready(stats)))
    _experiment_diagnostic_report(result, stats).to_csv(output_dir / "experiment_diagnostics.csv", index=False)
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
    _experiment_drawdown_episodes(result.backtest.equity_curve).to_csv(output_dir / "drawdown_episodes.csv", index=False)
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)


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
    (output_dir / "config.json").write_text(json_dump(config_payload))
    (output_dir / "summary.json").write_text(json_dump(json_ready(_sweep_summary_statistics(result))))
    result.table.to_csv(output_dir / "sweep.csv", index=False)
    _pareto_sweep_table(result.table).to_csv(output_dir / "pareto.csv", index=False)
    _parameter_summary_table(result).to_csv(output_dir / "parameter_summary.csv", index=False)
    _case_diagnostics(result).to_csv(output_dir / "case_diagnostics.csv", index=False)
    result.strategy_stats.to_csv(output_dir / "case_strategy_stats.csv", index=False)
    result.detector_stats.to_csv(output_dir / "case_detector_stats.csv", index=False)
    result.setup_stats.to_csv(output_dir / "case_setup_stats.csv", index=False)
    result.symbol_stats.to_csv(output_dir / "case_symbol_stats.csv", index=False)
    result.setup_order_decision_stats.to_csv(output_dir / "case_setup_order_decision_stats.csv", index=False)
    result.setup_strategy_filter_stats.to_csv(output_dir / "case_setup_strategy_filter_stats.csv", index=False)
    write_jsonl(output_dir / "case_configs.jsonl", sweep_case_config_records(result))
    result.data_inventory.to_csv(output_dir / "data_inventory.csv", index=False)
    result.data_coverage.to_csv(output_dir / "data_coverage.csv", index=False)
    result.limit_filter_audit.to_csv(output_dir / "limit_filter_audit.csv", index=False)
    symbol_metadata_for_config(result.config).to_csv(output_dir / "symbol_metadata.csv", index=False)


def _sweep_summary_statistics(result: PortfolioSweepResult | SingleStrategySweepResult) -> dict[str, object]:
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
        case_diagnostics=_case_diagnostics(result),
    )


def _pareto_sweep_table(table: pd.DataFrame) -> pd.DataFrame:
    return _build_pareto_sweep_table(table)


def _parameter_summary_table(result: PortfolioSweepResult | SingleStrategySweepResult) -> pd.DataFrame:
    return _build_parameter_summary_table(result.table, result.grid)


def _case_diagnostics(result: PortfolioSweepResult | SingleStrategySweepResult) -> pd.DataFrame:
    if not result.case_diagnostics.empty:
        return result.case_diagnostics
    return case_diagnostic_statistics(result.table)
