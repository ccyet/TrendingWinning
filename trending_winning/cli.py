from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
import sys

import pandas as pd

from trending_winning.backtest.experiment import (
    build_portfolio_benchmark_report,
    load_sweep_case_config,
    run_portfolio_parameter_sweep,
    run_portfolio_experiment,
    run_single_strategy_parameter_sweep,
    run_single_strategy_experiment,
    save_portfolio_benchmark,
)
from trending_winning.backtest.experiment_models import PortfolioExperimentConfig, SingleStrategyExperimentConfig
from trending_winning.backtest.engine import run_backtest
from trending_winning.backtest.models import BacktestConfig
from trending_winning.data.repository import MarketDataRepository
from trending_winning.data.tdx import diagnose_tdx_source
from trending_winning.data.tdx_parallels import (
    ParallelsTdxConfig,
    default_parallels_tdx_config,
    mac_path_to_parallels_shared_path,
    run_parallels_tdx_command,
)
from trending_winning.strategies.signal_bar import SUPPORTED_SIDE_MODES
from trending_winning.strategy import StrategyConfig, scan_bars

RISK_REWARD_HELP = (
    "盈亏比指向平仓信号，不决定开仓信号；开仓信号先给出开仓价和结构止损价，"
    "盈亏比只把风险距离换算成固定目标平仓价：多头=开仓价+(开仓价-止损价)*盈亏比，空头相反。"
)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_mapping(value: str) -> dict[str, float]:
    return {key: float(raw) for key, raw in _parse_key_value_pairs(value).items()}


def _parse_int_mapping(value: str) -> dict[str, int]:
    return {key: int(raw) for key, raw in _parse_key_value_pairs(value).items()}


def _parse_text_mapping(value: str) -> dict[str, str]:
    return _parse_key_value_pairs(value)


def _print_saved_artifact_manifest(output_dir: str | Path) -> None:
    path = Path(output_dir).expanduser() / "artifact_manifest.csv"
    print(f"artifact_manifest.csv saved: {path}")


def _artifact_manifest_table(
    output_dir: str | Path,
    *,
    category: str = "",
    max_priority: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_dir).expanduser()
    manifest_path = output_path / "artifact_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到产物索引：{manifest_path}")
    manifest = pd.read_csv(manifest_path)
    required = ["file_name", "category", "priority", "question", "description"]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"artifact_manifest.csv 缺少字段：{', '.join(missing)}")
    table = manifest.loc[:, required].copy()
    table["priority"] = pd.to_numeric(table["priority"], errors="coerce").fillna(99).astype(int)
    if category:
        table = table.loc[table["category"].astype(str).eq(category)]
    if max_priority is not None:
        table = table.loc[table["priority"].le(max_priority)]
    table = table.sort_values(["priority", "category", "file_name"], kind="stable").reset_index(drop=True)
    table["path"] = [str(output_path / str(file_name)) for file_name in table["file_name"]]
    return table.rename(
        columns={
            "file_name": "文件",
            "category": "类别",
            "priority": "优先级",
            "question": "先回答的问题",
            "description": "说明",
            "path": "本机路径",
        }
    )


def _add_side_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--side-mode",
        choices=list(SUPPORTED_SIDE_MODES),
        default="both",
        help="订单方向：both=多空都做，long_only=仅做多，short_only=仅做空。",
    )


def _add_trailing_take_profit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trailing-take-profit-activation-pct",
        type=float,
        default=0.0,
        help="回撤止盈可选启动浮盈，小数比例；按上一根已完成 K 确认，0 表示不设门槛。",
    )
    parser.add_argument(
        "--trailing-take-profit-drawdown-pct",
        type=float,
        default=0.0,
        help="比例止盈参数，也就是最大盈利回撤幅度；按上一根已完成 K 的最大盈利价位计算平仓线，例如多头最高浮盈后回撤到线即平仓，0 表示关闭。",
    )
    parser.add_argument(
        "--trailing-take-profit-ma-period",
        type=int,
        default=0,
        help="均线回撤止盈当前周期均线周期；由用户输入 K 数，按当前周期上一根已完成 K 的均线触发，独立于比例止盈，0 表示关闭。",
    )


def _parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_terminal_false_breakout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--enable-terminal-false-breakout-filter",
        action="store_true",
        help="启用末端假突破过滤；只在策略层过滤开仓订单，不改变形态识别和撮合逻辑。",
    )
    parser.add_argument(
        "--terminal-false-breakout-detectors",
        default="trend,channel",
        help="末端假突破过滤适用的形态模块，逗号分隔；默认 trend,channel。",
    )
    parser.add_argument("--terminal-false-breakout-lookback", type=int, default=40)
    parser.add_argument("--terminal-false-breakout-atr-period", type=int, default=14)
    parser.add_argument("--terminal-false-breakout-min-regime-bars", type=int, default=18)
    parser.add_argument("--terminal-false-breakout-extension-atr-multiple", type=float, default=2.0)
    parser.add_argument("--terminal-false-breakout-edge-lookback", type=int, default=8)
    parser.add_argument("--terminal-false-breakout-edge-pos", type=float, default=0.90)
    parser.add_argument("--terminal-false-breakout-edge-min-count", type=int, default=3)
    parser.add_argument("--terminal-false-breakout-weak-progress-atr", type=float, default=0.35)
    parser.add_argument("--terminal-false-breakout-wick-ratio", type=float, default=0.35)
    parser.add_argument("--terminal-false-breakout-min-score", type=int, default=3)


def _terminal_false_breakout_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "terminal_false_breakout_enabled": bool(args.enable_terminal_false_breakout_filter),
        "terminal_false_breakout_detectors": _parse_csv_tuple(str(args.terminal_false_breakout_detectors)),
        "terminal_false_breakout_lookback": int(args.terminal_false_breakout_lookback),
        "terminal_false_breakout_atr_period": int(args.terminal_false_breakout_atr_period),
        "terminal_false_breakout_min_regime_bars": int(args.terminal_false_breakout_min_regime_bars),
        "terminal_false_breakout_extension_atr_multiple": float(
            args.terminal_false_breakout_extension_atr_multiple
        ),
        "terminal_false_breakout_edge_lookback": int(args.terminal_false_breakout_edge_lookback),
        "terminal_false_breakout_edge_pos": float(args.terminal_false_breakout_edge_pos),
        "terminal_false_breakout_edge_min_count": int(args.terminal_false_breakout_edge_min_count),
        "terminal_false_breakout_weak_progress_atr": float(args.terminal_false_breakout_weak_progress_atr),
        "terminal_false_breakout_wick_ratio": float(args.terminal_false_breakout_wick_ratio),
        "terminal_false_breakout_min_score": int(args.terminal_false_breakout_min_score),
    }


def _parse_generic_sweep_grid(items: list[str], config: object) -> dict[str, list[object]]:
    """解析通用参数网格；字段合法性由实验配置 dataclass 决定。"""
    if not items:
        return {}
    field_names = {field.name for field in fields(type(config))}
    grid: dict[str, list[object]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--grid 必须使用 field=value1,value2 格式：{item}")
        key, raw_values = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--grid 字段和值不能为空：{item}")
        if key not in field_names:
            raise ValueError(f"--grid 不支持配置字段：{key}")
        current_value = getattr(config, key)
        values_text = _split_grid_values(raw_values, current_value)
        if not values_text:
            raise ValueError(f"--grid 字段和值不能为空：{item}")
        grid[key] = [_parse_grid_value(value, current_value) for value in values_text]
    return grid


def _split_grid_values(raw_values: str, current_value: object) -> list[str]:
    separator = ";" if isinstance(current_value, Mapping) else ","
    return [value.strip() for value in raw_values.split(separator) if value.strip()]


def _parse_grid_value(value: str, current_value: object) -> object:
    if isinstance(current_value, bool):
        return _parse_bool_value(value)
    if isinstance(current_value, int):
        return int(value)
    if isinstance(current_value, float):
        return float(value)
    if isinstance(current_value, Mapping):
        return _parse_grid_mapping(value)
    if isinstance(current_value, tuple):
        return tuple(part.strip() for part in value.split("+") if part.strip())
    if current_value is None:
        return _infer_grid_scalar(value)
    return value


def _parse_grid_mapping(value: str) -> dict[str, object]:
    """解析 mapping 型 grid 单元；多个键值用 + 连接，多个方案由上层 ; 分隔。"""
    normalized = value.strip()
    if normalized == "{}":
        return {}
    result: dict[str, object] = {}
    for item in normalized.split("+"):
        text = item.strip()
        if not text:
            continue
        if "=" in text:
            key, raw = text.split("=", 1)
        elif ":" in text:
            key, raw = text.split(":", 1)
        else:
            raise ValueError(f"mapping grid 必须使用 key=value 或 key:value：{text}")
        key = key.strip()
        if not key:
            raise ValueError(f"mapping grid 的 key 不能为空：{text}")
        result[key] = _infer_grid_scalar(raw.strip())
    return result


def _parse_bool_value(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"布尔参数只支持 true/false：{value}")


def _infer_grid_scalar(value: str) -> object:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    if normalized in {"true", "false"}:
        return _parse_bool_value(normalized)
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_key_value_pairs(value: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"映射参数必须使用 key=value 格式：{text}")
        key, raw = text.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if not key or not raw:
            raise ValueError(f"映射参数不能为空：{text}")
        pairs[key] = raw
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="TDX-only trend strategy toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="fetch minute bars from TDX")
    fetch_parser.add_argument("--symbols", required=True)
    fetch_parser.add_argument("--timeframe", required=True, choices=["1d", "5m", "15m", "30m", "60m"])
    fetch_parser.add_argument("--start", required=True)
    fetch_parser.add_argument("--end", required=True)
    fetch_parser.add_argument("--adjust", default="qfq")
    fetch_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    fetch_parser.add_argument("--tdx-path", default="")
    _add_tdx_runtime_args(fetch_parser)

    doctor_parser = subparsers.add_parser("tdx-doctor", help="diagnose TDX import, login and sample K-line requests")
    doctor_parser.add_argument("--symbols", required=True)
    doctor_parser.add_argument("--timeframes", default="1d,5m,15m,30m,60m")
    doctor_parser.add_argument("--start", required=True)
    doctor_parser.add_argument("--end", required=True)
    doctor_parser.add_argument("--adjust", default="qfq")
    doctor_parser.add_argument("--tdx-path", default="")
    _add_tdx_runtime_args(doctor_parser)

    prepare_parser = subparsers.add_parser("prepare-data", help="audit local data and fetch only missing/bad bars from TDX")
    prepare_parser.add_argument("--symbols", required=True)
    prepare_parser.add_argument("--timeframes", required=True)
    prepare_parser.add_argument("--start", required=True)
    prepare_parser.add_argument("--end", required=True)
    prepare_parser.add_argument("--adjust", default="qfq")
    prepare_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    prepare_parser.add_argument("--tdx-path", default="")
    prepare_parser.add_argument("--min-coverage-ratio", type=float, default=None)
    prepare_parser.add_argument("--allow-incomplete-after-update", action="store_true")
    _add_tdx_runtime_args(prepare_parser)

    plan_parser = subparsers.add_parser("plan-data", help="audit local data and print the TDX fetch plan")
    plan_parser.add_argument("--symbols", required=True)
    plan_parser.add_argument("--timeframes", required=True)
    plan_parser.add_argument("--start", required=True)
    plan_parser.add_argument("--end", required=True)
    plan_parser.add_argument("--adjust", default="qfq")
    plan_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    plan_parser.add_argument("--min-coverage-ratio", type=float, default=None)

    audit_parser = subparsers.add_parser("audit-data", help="audit local parquet coverage before backtesting")
    audit_parser.add_argument("--symbols", required=True)
    audit_parser.add_argument("--timeframe", required=True, choices=["1d", "5m", "15m", "30m", "60m"])
    audit_parser.add_argument("--higher-timeframe", default="", choices=["", "5m", "15m", "30m", "60m"])
    audit_parser.add_argument("--higher-timeframe-max-age-minutes", type=int, default=None)
    audit_parser.add_argument("--start", required=True)
    audit_parser.add_argument("--end", required=True)
    audit_parser.add_argument("--adjust", default="qfq")
    audit_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    audit_parser.add_argument(
        "--show-gap-episodes",
        action="store_true",
        help="同时输出每段连续缺失 K 的起止时间、缺失根数和前后可用 K。",
    )

    inventory_parser = subparsers.add_parser("inventory-data", help="list local parquet cache inventory by timeframe")
    inventory_parser.add_argument("--symbols", default="")
    inventory_parser.add_argument("--timeframes", default="1d,5m,15m,30m,60m")
    inventory_parser.add_argument("--adjust", default="qfq")
    inventory_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")

    artifacts_parser = subparsers.add_parser("show-artifacts", help="print a saved run artifact_manifest.csv")
    artifacts_parser.add_argument("--output-dir", required=True)
    artifacts_parser.add_argument("--category", default="", help="只显示指定类别，例如 数据质量、订单与过滤。")
    artifacts_parser.add_argument("--max-priority", type=int, default=None, help="只显示不高于该优先级的产物。")

    scan_parser = subparsers.add_parser("backtest", help="scan local bars and run a breakout backtest")
    scan_parser.add_argument("--symbols", required=True)
    scan_parser.add_argument("--timeframe", required=True, choices=["5m", "15m", "30m", "60m"])
    scan_parser.add_argument("--start", required=True)
    scan_parser.add_argument("--end", required=True)
    scan_parser.add_argument("--adjust", default="qfq")
    scan_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")

    single_parser = subparsers.add_parser("single-backtest", help="run one detector strategy without portfolio allocation")
    single_parser.add_argument("--symbols", required=True)
    single_parser.add_argument("--timeframe", required=True, choices=["5m", "15m", "30m", "60m"])
    single_parser.add_argument("--higher-timeframe", default="", choices=["", "5m", "15m", "30m", "60m"])
    single_parser.add_argument("--higher-timeframe-max-age-minutes", type=int, default=None)
    single_parser.add_argument("--start", required=True)
    single_parser.add_argument("--end", required=True)
    single_parser.add_argument("--adjust", default="qfq")
    single_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    single_parser.add_argument("--detector", required=True, choices=["trend", "range", "channel", "reversal"])
    single_parser.add_argument("--risk-reward", type=float, default=2.0, help=RISK_REWARD_HELP)
    single_parser.add_argument("--max-holding-bars", type=int, default=12)
    single_parser.add_argument("--max-actual-risk-pct", type=float, default=None)
    single_parser.add_argument("--max-chase-pct", type=float, default=None)
    _add_side_mode_arg(single_parser)
    single_parser.add_argument("--min-coverage-ratio", type=float, default=None)
    single_parser.add_argument("--fee-rate", type=float, default=0.0)
    single_parser.add_argument("--slippage-bps", type=float, default=0.0)
    single_parser.add_argument("--initial-equity", type=float, default=1.0)
    single_parser.add_argument("--intrabar-exit-policy", choices=["conservative", "optimistic"], default="conservative")
    _add_trailing_take_profit_args(single_parser)
    _add_terminal_false_breakout_args(single_parser)
    single_parser.add_argument("--allow-bad-data", action="store_true")
    single_parser.add_argument("--output-dir", default="")
    single_parser.add_argument("--trend-lookback", type=int, default=20)
    single_parser.add_argument("--trend-min-score", type=float, default=1.0)
    single_parser.add_argument("--trend-strong-close-pos", type=float, default=0.65)
    single_parser.add_argument("--trend-min-body-ratio", type=float, default=0.45)
    single_parser.add_argument("--trend-pullback-lookback", type=int, default=5)
    single_parser.add_argument("--trend-h2-min-pullback-legs", type=int, default=2)
    single_parser.add_argument("--range-lookback", type=int, default=20)
    single_parser.add_argument("--range-middle-low", type=float, default=0.25)
    single_parser.add_argument("--range-middle-high", type=float, default=0.75)
    single_parser.add_argument("--range-false-break-buffer", type=float, default=0.0)
    single_parser.add_argument("--range-strong-close-pos", type=float, default=0.65)
    single_parser.add_argument("--range-min-score", type=float, default=0.8)
    single_parser.add_argument("--channel-method", choices=["regression", "swing"], default="regression")
    single_parser.add_argument("--channel-lookback", type=int, default=40)
    single_parser.add_argument("--channel-sigma-multiple", type=float, default=2.0)
    single_parser.add_argument("--channel-break-buffer", type=float, default=0.0)
    single_parser.add_argument("--channel-swing-left-bars", type=int, default=2)
    single_parser.add_argument("--channel-swing-right-bars", type=int, default=2)
    single_parser.add_argument("--reversal-lookback", type=int, default=20)
    single_parser.add_argument("--reversal-strong-close-pos", type=float, default=0.65)
    single_parser.add_argument("--reversal-min-body-ratio", type=float, default=0.45)
    single_parser.add_argument("--reversal-old-extreme-tolerance-pct", type=float, default=0.01)
    single_parser.add_argument("--disable-reversal-old-extreme-test", action="store_true")
    single_parser.add_argument("--disable-reversal-structure-confirmation", action="store_true")

    single_sweep_parser = subparsers.add_parser("single-sweep", help="run one detector parameter grid without portfolio allocation")
    single_sweep_parser.add_argument("--symbols", required=True)
    single_sweep_parser.add_argument("--timeframe", required=True, choices=["5m", "15m", "30m", "60m"])
    single_sweep_parser.add_argument("--higher-timeframe", default="", choices=["", "5m", "15m", "30m", "60m"])
    single_sweep_parser.add_argument("--higher-timeframe-max-age-minutes", type=int, default=None)
    single_sweep_parser.add_argument("--start", required=True)
    single_sweep_parser.add_argument("--end", required=True)
    single_sweep_parser.add_argument("--adjust", default="qfq")
    single_sweep_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    single_sweep_parser.add_argument("--detector", required=True, choices=["trend", "range", "channel", "reversal"])
    single_sweep_parser.add_argument("--risk-rewards", required=True, help=RISK_REWARD_HELP)
    single_sweep_parser.add_argument("--max-holding-bars-list", required=True)
    single_sweep_parser.add_argument("--trend-min-scores", default="")
    single_sweep_parser.add_argument("--grid", action="append", default=[])
    single_sweep_parser.add_argument("--max-actual-risk-pct", type=float, default=None)
    single_sweep_parser.add_argument("--max-chase-pct", type=float, default=None)
    _add_side_mode_arg(single_sweep_parser)
    single_sweep_parser.add_argument("--min-coverage-ratio", type=float, default=None)
    single_sweep_parser.add_argument("--fee-rate", type=float, default=0.0)
    single_sweep_parser.add_argument("--slippage-bps", type=float, default=0.0)
    single_sweep_parser.add_argument("--initial-equity", type=float, default=1.0)
    single_sweep_parser.add_argument("--intrabar-exit-policy", choices=["conservative", "optimistic"], default="conservative")
    _add_trailing_take_profit_args(single_sweep_parser)
    _add_terminal_false_breakout_args(single_sweep_parser)
    single_sweep_parser.add_argument("--allow-bad-data", action="store_true")
    single_sweep_parser.add_argument("--output-dir", default="")
    single_sweep_parser.add_argument("--trend-lookback", type=int, default=20)
    single_sweep_parser.add_argument("--trend-min-score", type=float, default=1.0)
    single_sweep_parser.add_argument("--trend-strong-close-pos", type=float, default=0.65)
    single_sweep_parser.add_argument("--trend-min-body-ratio", type=float, default=0.45)
    single_sweep_parser.add_argument("--trend-pullback-lookback", type=int, default=5)
    single_sweep_parser.add_argument("--trend-h2-min-pullback-legs", type=int, default=2)
    single_sweep_parser.add_argument("--range-lookback", type=int, default=20)
    single_sweep_parser.add_argument("--range-middle-low", type=float, default=0.25)
    single_sweep_parser.add_argument("--range-middle-high", type=float, default=0.75)
    single_sweep_parser.add_argument("--range-false-break-buffer", type=float, default=0.0)
    single_sweep_parser.add_argument("--range-strong-close-pos", type=float, default=0.65)
    single_sweep_parser.add_argument("--range-min-score", type=float, default=0.8)
    single_sweep_parser.add_argument("--channel-method", choices=["regression", "swing"], default="regression")
    single_sweep_parser.add_argument("--channel-lookback", type=int, default=40)
    single_sweep_parser.add_argument("--channel-sigma-multiple", type=float, default=2.0)
    single_sweep_parser.add_argument("--channel-break-buffer", type=float, default=0.0)
    single_sweep_parser.add_argument("--channel-swing-left-bars", type=int, default=2)
    single_sweep_parser.add_argument("--channel-swing-right-bars", type=int, default=2)
    single_sweep_parser.add_argument("--reversal-lookback", type=int, default=20)
    single_sweep_parser.add_argument("--reversal-strong-close-pos", type=float, default=0.65)
    single_sweep_parser.add_argument("--reversal-min-body-ratio", type=float, default=0.45)
    single_sweep_parser.add_argument("--reversal-old-extreme-tolerance-pct", type=float, default=0.01)
    single_sweep_parser.add_argument("--disable-reversal-old-extreme-test", action="store_true")
    single_sweep_parser.add_argument("--disable-reversal-structure-confirmation", action="store_true")

    portfolio_parser = subparsers.add_parser("portfolio-backtest", help="run independent detector strategies as a portfolio")
    portfolio_parser.add_argument("--symbols", required=True)
    portfolio_parser.add_argument("--timeframe", required=True, choices=["5m", "15m", "30m", "60m"])
    portfolio_parser.add_argument("--higher-timeframe", default="", choices=["", "5m", "15m", "30m", "60m"])
    portfolio_parser.add_argument("--higher-timeframe-max-age-minutes", type=int, default=None)
    portfolio_parser.add_argument("--start", required=True)
    portfolio_parser.add_argument("--end", required=True)
    portfolio_parser.add_argument("--adjust", default="qfq")
    portfolio_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    portfolio_parser.add_argument("--detectors", default="trend,range,channel")
    portfolio_parser.add_argument("--risk-reward", type=float, default=2.0, help=RISK_REWARD_HELP)
    portfolio_parser.add_argument("--max-holding-bars", type=int, default=12)
    portfolio_parser.add_argument("--max-actual-risk-pct", type=float, default=None)
    portfolio_parser.add_argument("--max-chase-pct", type=float, default=None)
    _add_side_mode_arg(portfolio_parser)
    portfolio_parser.add_argument("--min-coverage-ratio", type=float, default=None)
    portfolio_parser.add_argument("--fee-rate", type=float, default=0.0)
    portfolio_parser.add_argument("--slippage-bps", type=float, default=0.0)
    portfolio_parser.add_argument("--initial-equity", type=float, default=1.0)
    portfolio_parser.add_argument("--max-open-positions", type=int, default=5)
    portfolio_parser.add_argument("--capital-per-trade", type=float, default=None)
    portfolio_parser.add_argument("--risk-per-trade", type=float, default=None)
    portfolio_parser.add_argument("--max-capital-per-trade", type=float, default=1.0)
    portfolio_parser.add_argument("--short-margin-rate", type=float, default=1.0)
    portfolio_parser.add_argument("--reserve-cash", type=float, default=0.0)
    portfolio_parser.add_argument("--allow-same-symbol-overlap", action="store_true")
    portfolio_parser.add_argument("--strategy-priority", default="")
    portfolio_parser.add_argument("--strategy-capital-limit", default="")
    portfolio_parser.add_argument("--sector-capital-limit", default="")
    portfolio_parser.add_argument("--symbol-sector-map", default="")
    portfolio_parser.add_argument("--sector-metadata-key", default="sector")
    portfolio_parser.add_argument("--default-sector", default="UNKNOWN")
    portfolio_parser.add_argument("--intrabar-exit-policy", choices=["conservative", "optimistic"], default="conservative")
    _add_trailing_take_profit_args(portfolio_parser)
    _add_terminal_false_breakout_args(portfolio_parser)
    portfolio_parser.add_argument("--trend-lookback", type=int, default=20)
    portfolio_parser.add_argument("--trend-min-score", type=float, default=1.0)
    portfolio_parser.add_argument("--trend-strong-close-pos", type=float, default=0.65)
    portfolio_parser.add_argument("--trend-min-body-ratio", type=float, default=0.45)
    portfolio_parser.add_argument("--trend-pullback-lookback", type=int, default=5)
    portfolio_parser.add_argument("--trend-h2-min-pullback-legs", type=int, default=2)
    portfolio_parser.add_argument("--range-lookback", type=int, default=20)
    portfolio_parser.add_argument("--range-middle-low", type=float, default=0.25)
    portfolio_parser.add_argument("--range-middle-high", type=float, default=0.75)
    portfolio_parser.add_argument("--range-false-break-buffer", type=float, default=0.0)
    portfolio_parser.add_argument("--range-strong-close-pos", type=float, default=0.65)
    portfolio_parser.add_argument("--range-min-score", type=float, default=0.8)
    portfolio_parser.add_argument("--channel-method", choices=["regression", "swing"], default="regression")
    portfolio_parser.add_argument("--channel-lookback", type=int, default=40)
    portfolio_parser.add_argument("--channel-sigma-multiple", type=float, default=2.0)
    portfolio_parser.add_argument("--channel-break-buffer", type=float, default=0.0)
    portfolio_parser.add_argument("--channel-swing-left-bars", type=int, default=2)
    portfolio_parser.add_argument("--channel-swing-right-bars", type=int, default=2)
    portfolio_parser.add_argument("--reversal-lookback", type=int, default=20)
    portfolio_parser.add_argument("--reversal-strong-close-pos", type=float, default=0.65)
    portfolio_parser.add_argument("--reversal-min-body-ratio", type=float, default=0.45)
    portfolio_parser.add_argument("--reversal-old-extreme-tolerance-pct", type=float, default=0.01)
    portfolio_parser.add_argument("--disable-reversal-old-extreme-test", action="store_true")
    portfolio_parser.add_argument("--disable-reversal-structure-confirmation", action="store_true")
    portfolio_parser.add_argument("--allow-bad-data", action="store_true")
    portfolio_parser.add_argument("--output-dir", default="")
    portfolio_parser.add_argument("--benchmark", action="store_true")

    sweep_parser = subparsers.add_parser("portfolio-sweep", help="run a parameter grid with one shared data load")
    sweep_parser.add_argument("--symbols", required=True)
    sweep_parser.add_argument("--timeframe", required=True, choices=["5m", "15m", "30m", "60m"])
    sweep_parser.add_argument("--higher-timeframe", default="", choices=["", "5m", "15m", "30m", "60m"])
    sweep_parser.add_argument("--higher-timeframe-max-age-minutes", type=int, default=None)
    sweep_parser.add_argument("--start", required=True)
    sweep_parser.add_argument("--end", required=True)
    sweep_parser.add_argument("--adjust", default="qfq")
    sweep_parser.add_argument("--data-root", default="/Users/a1234/Desktop/trend-backtest/data/market/daily")
    sweep_parser.add_argument("--detectors", default="trend,range,channel")
    sweep_parser.add_argument("--risk-rewards", required=True, help=RISK_REWARD_HELP)
    sweep_parser.add_argument("--max-holding-bars-list", required=True)
    sweep_parser.add_argument("--grid", action="append", default=[])
    sweep_parser.add_argument("--max-actual-risk-pct", type=float, default=None)
    sweep_parser.add_argument("--max-chase-pct", type=float, default=None)
    _add_side_mode_arg(sweep_parser)
    sweep_parser.add_argument("--min-coverage-ratio", type=float, default=None)
    sweep_parser.add_argument("--fee-rate", type=float, default=0.0)
    sweep_parser.add_argument("--slippage-bps", type=float, default=0.0)
    sweep_parser.add_argument("--initial-equity", type=float, default=1.0)
    sweep_parser.add_argument("--max-open-positions-list", default="5")
    sweep_parser.add_argument("--capital-per-trade", type=float, default=None)
    sweep_parser.add_argument("--risk-per-trade", type=float, default=None)
    sweep_parser.add_argument("--max-capital-per-trade", type=float, default=1.0)
    sweep_parser.add_argument("--short-margin-rate", type=float, default=1.0)
    sweep_parser.add_argument("--reserve-cash", type=float, default=0.0)
    sweep_parser.add_argument("--allow-same-symbol-overlap", action="store_true")
    sweep_parser.add_argument("--strategy-priority", default="")
    sweep_parser.add_argument("--strategy-capital-limit", default="")
    sweep_parser.add_argument("--sector-capital-limit", default="")
    sweep_parser.add_argument("--symbol-sector-map", default="")
    sweep_parser.add_argument("--sector-metadata-key", default="sector")
    sweep_parser.add_argument("--default-sector", default="UNKNOWN")
    sweep_parser.add_argument("--intrabar-exit-policy", choices=["conservative", "optimistic"], default="conservative")
    _add_trailing_take_profit_args(sweep_parser)
    _add_terminal_false_breakout_args(sweep_parser)
    sweep_parser.add_argument("--trend-lookback", type=int, default=20)
    sweep_parser.add_argument("--trend-min-score", type=float, default=1.0)
    sweep_parser.add_argument("--trend-strong-close-pos", type=float, default=0.65)
    sweep_parser.add_argument("--trend-min-body-ratio", type=float, default=0.45)
    sweep_parser.add_argument("--trend-pullback-lookback", type=int, default=5)
    sweep_parser.add_argument("--trend-h2-min-pullback-legs", type=int, default=2)
    sweep_parser.add_argument("--range-lookback", type=int, default=20)
    sweep_parser.add_argument("--range-middle-low", type=float, default=0.25)
    sweep_parser.add_argument("--range-middle-high", type=float, default=0.75)
    sweep_parser.add_argument("--range-false-break-buffer", type=float, default=0.0)
    sweep_parser.add_argument("--range-strong-close-pos", type=float, default=0.65)
    sweep_parser.add_argument("--range-min-score", type=float, default=0.8)
    sweep_parser.add_argument("--channel-method", choices=["regression", "swing"], default="regression")
    sweep_parser.add_argument("--channel-lookback", type=int, default=40)
    sweep_parser.add_argument("--channel-sigma-multiple", type=float, default=2.0)
    sweep_parser.add_argument("--channel-break-buffer", type=float, default=0.0)
    sweep_parser.add_argument("--channel-swing-left-bars", type=int, default=2)
    sweep_parser.add_argument("--channel-swing-right-bars", type=int, default=2)
    sweep_parser.add_argument("--reversal-lookback", type=int, default=20)
    sweep_parser.add_argument("--reversal-strong-close-pos", type=float, default=0.65)
    sweep_parser.add_argument("--reversal-min-body-ratio", type=float, default=0.45)
    sweep_parser.add_argument("--reversal-old-extreme-tolerance-pct", type=float, default=0.01)
    sweep_parser.add_argument("--disable-reversal-old-extreme-test", action="store_true")
    sweep_parser.add_argument("--disable-reversal-structure-confirmation", action="store_true")
    sweep_parser.add_argument("--allow-bad-data", action="store_true")
    sweep_parser.add_argument("--output-dir", default="")

    replay_parser = subparsers.add_parser("replay-case", help="replay one case from a saved case_configs.jsonl")
    replay_parser.add_argument("--case-configs", required=True)
    replay_parser.add_argument("--case-config-hash", default="")
    replay_parser.add_argument("--case-name", default="")
    replay_parser.add_argument("--output-dir", default="")

    args = parser.parse_args()

    if args.command == "show-artifacts":
        print(
            _artifact_manifest_table(
                args.output_dir,
                category=str(args.category).strip(),
                max_priority=args.max_priority,
            ).to_string(index=False)
        )
        return

    if args.command == "replay-case":
        config = load_sweep_case_config(
            args.case_configs,
            case_config_hash=str(args.case_config_hash).strip(),
            case_name=str(args.case_name).strip(),
        )
        if args.output_dir:
            config = replace(config, output_dir=str(args.output_dir))
        save = bool(args.output_dir)
        if isinstance(config, SingleStrategyExperimentConfig):
            experiment = run_single_strategy_experiment(config, save=save)
        else:
            experiment = run_portfolio_experiment(config, save=save)
        case_key = str(args.case_config_hash or args.case_name).strip()
        print(f"replayed case: {case_key}")
        print(experiment.backtest.stats)
        if args.output_dir:
            print(f"replay output saved: {Path(args.output_dir).expanduser()}")
            _print_saved_artifact_manifest(args.output_dir)
        return

    symbols = tuple(item.strip() for item in args.symbols.split(",") if item.strip())
    if args.command in {"tdx-doctor", "fetch", "prepare-data"} and _resolve_tdx_runtime(args.runtime) == "parallels":
        _run_tdx_cli_in_parallels(args)
        return
    if args.command == "tdx-doctor":
        result = diagnose_tdx_source(
            symbols=symbols,
            timeframes=tuple(item.strip() for item in args.timeframes.split(",") if item.strip()),
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            tqcenter_path=args.tdx_path,
        )
        print(result.to_string(index=False))
        return

    repo = MarketDataRepository(Path(args.data_root), adjust=args.adjust)
    if args.command == "inventory-data":
        result = repo.inventory(
            timeframes=tuple(item.strip() for item in args.timeframes.split(",") if item.strip()),
            symbols=symbols or None,
        )
        print(result.to_string(index=False))
        return

    if args.command == "fetch":
        result = repo.update_from_tdx(
            symbols=symbols,
            start=args.start,
            end=args.end,
            timeframe=args.timeframe,
            tqcenter_path=args.tdx_path,
        )
        print(result.to_string(index=False))
        return

    if args.command == "prepare-data":
        result = repo.prepare_from_tdx(
            symbols=symbols,
            timeframes=tuple(item.strip() for item in args.timeframes.split(",") if item.strip()),
            start=args.start,
            end=args.end,
            tqcenter_path=args.tdx_path,
            min_coverage_ratio=args.min_coverage_ratio,
            strict_after_update=not bool(args.allow_incomplete_after_update),
        )
        print(result.to_string(index=False))
        return

    if args.command == "plan-data":
        result = repo.plan_from_tdx(
            symbols=symbols,
            timeframes=tuple(item.strip() for item in args.timeframes.split(",") if item.strip()),
            start=args.start,
            end=args.end,
            min_coverage_ratio=args.min_coverage_ratio,
        )
        print(result.to_string(index=False))
        return

    if args.command == "audit-data":
        timeframes = [str(args.timeframe)]
        higher_timeframe = str(args.higher_timeframe).strip()
        if higher_timeframe and higher_timeframe != args.timeframe:
            timeframes.append(higher_timeframe)
        frames = [
            repo.audit_bars(
                timeframe=timeframe,
                symbols=symbols,
                start=args.start,
                end=args.end,
            )
            for timeframe in timeframes
        ]
        result = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        print(result.to_string(index=False))
        if args.show_gap_episodes:
            gap_frames = [
                repo.data_gap_episodes(
                    timeframe=timeframe,
                    symbols=symbols,
                    start=args.start,
                    end=args.end,
                )
                for timeframe in timeframes
            ]
            gaps = pd.concat(gap_frames, ignore_index=True) if len(gap_frames) > 1 else gap_frames[0]
            print("\n数据缺口明细")
            print("无连续缺口" if gaps.empty else gaps.to_string(index=False))
        return

    if args.command == "single-backtest":
        config = SingleStrategyExperimentConfig(
            name="single-backtest",
            data_root=str(args.data_root),
            symbols=symbols,
            timeframe=args.timeframe,
            higher_timeframe=str(args.higher_timeframe),
            higher_timeframe_max_age_minutes=args.higher_timeframe_max_age_minutes,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            detector=str(args.detector),
            risk_reward=float(args.risk_reward),
            max_holding_bars=int(args.max_holding_bars),
            max_actual_risk_pct=args.max_actual_risk_pct,
            max_chase_pct=args.max_chase_pct,
            side_mode=str(args.side_mode),
            min_coverage_ratio=args.min_coverage_ratio,
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            initial_equity=float(args.initial_equity),
            intrabar_exit_policy=str(args.intrabar_exit_policy),
            trailing_take_profit_activation_pct=float(args.trailing_take_profit_activation_pct),
            trailing_take_profit_drawdown_pct=float(args.trailing_take_profit_drawdown_pct),
            trailing_take_profit_ma_period=int(args.trailing_take_profit_ma_period),
            **_terminal_false_breakout_kwargs(args),
            strict_data_quality=not bool(args.allow_bad_data),
            output_dir=args.output_dir,
            trend_lookback=int(args.trend_lookback),
            trend_min_score=float(args.trend_min_score),
            trend_strong_close_pos=float(args.trend_strong_close_pos),
            trend_min_body_ratio=float(args.trend_min_body_ratio),
            trend_pullback_lookback=int(args.trend_pullback_lookback),
            trend_h2_min_pullback_legs=int(args.trend_h2_min_pullback_legs),
            range_lookback=int(args.range_lookback),
            range_middle_low=float(args.range_middle_low),
            range_middle_high=float(args.range_middle_high),
            range_false_break_buffer=float(args.range_false_break_buffer),
            range_strong_close_pos=float(args.range_strong_close_pos),
            range_min_score=float(args.range_min_score),
            channel_method=str(args.channel_method),
            channel_lookback=int(args.channel_lookback),
            channel_sigma_multiple=float(args.channel_sigma_multiple),
            channel_break_buffer=float(args.channel_break_buffer),
            channel_swing_left_bars=int(args.channel_swing_left_bars),
            channel_swing_right_bars=int(args.channel_swing_right_bars),
            reversal_lookback=int(args.reversal_lookback),
            reversal_strong_close_pos=float(args.reversal_strong_close_pos),
            reversal_min_body_ratio=float(args.reversal_min_body_ratio),
            reversal_old_extreme_tolerance_pct=float(args.reversal_old_extreme_tolerance_pct),
            reversal_require_old_extreme_test=not bool(args.disable_reversal_old_extreme_test),
            reversal_require_structure_confirmation=not bool(args.disable_reversal_structure_confirmation),
        )
        experiment = run_single_strategy_experiment(config, save=bool(args.output_dir))
        result = experiment.backtest
        print(result.stats)
        if not result.equity_curve.empty:
            print(result.equity_curve.tail(20).to_string(index=False))
        if not result.trades.empty:
            print(result.trades.to_string(index=False))
        if args.output_dir:
            _print_saved_artifact_manifest(args.output_dir)
        return

    if args.command == "single-sweep":
        config = SingleStrategyExperimentConfig(
            name="single-sweep",
            data_root=str(args.data_root),
            symbols=symbols,
            timeframe=args.timeframe,
            higher_timeframe=str(args.higher_timeframe),
            higher_timeframe_max_age_minutes=args.higher_timeframe_max_age_minutes,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            detector=str(args.detector),
            max_actual_risk_pct=args.max_actual_risk_pct,
            max_chase_pct=args.max_chase_pct,
            side_mode=str(args.side_mode),
            min_coverage_ratio=args.min_coverage_ratio,
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            initial_equity=float(args.initial_equity),
            intrabar_exit_policy=str(args.intrabar_exit_policy),
            trailing_take_profit_activation_pct=float(args.trailing_take_profit_activation_pct),
            trailing_take_profit_drawdown_pct=float(args.trailing_take_profit_drawdown_pct),
            trailing_take_profit_ma_period=int(args.trailing_take_profit_ma_period),
            **_terminal_false_breakout_kwargs(args),
            strict_data_quality=not bool(args.allow_bad_data),
            output_dir=args.output_dir,
            trend_lookback=int(args.trend_lookback),
            trend_min_score=float(args.trend_min_score),
            trend_strong_close_pos=float(args.trend_strong_close_pos),
            trend_min_body_ratio=float(args.trend_min_body_ratio),
            trend_pullback_lookback=int(args.trend_pullback_lookback),
            trend_h2_min_pullback_legs=int(args.trend_h2_min_pullback_legs),
            range_lookback=int(args.range_lookback),
            range_middle_low=float(args.range_middle_low),
            range_middle_high=float(args.range_middle_high),
            range_false_break_buffer=float(args.range_false_break_buffer),
            range_strong_close_pos=float(args.range_strong_close_pos),
            range_min_score=float(args.range_min_score),
            channel_method=str(args.channel_method),
            channel_lookback=int(args.channel_lookback),
            channel_sigma_multiple=float(args.channel_sigma_multiple),
            channel_break_buffer=float(args.channel_break_buffer),
            channel_swing_left_bars=int(args.channel_swing_left_bars),
            channel_swing_right_bars=int(args.channel_swing_right_bars),
            reversal_lookback=int(args.reversal_lookback),
            reversal_strong_close_pos=float(args.reversal_strong_close_pos),
            reversal_min_body_ratio=float(args.reversal_min_body_ratio),
            reversal_old_extreme_tolerance_pct=float(args.reversal_old_extreme_tolerance_pct),
            reversal_require_old_extreme_test=not bool(args.disable_reversal_old_extreme_test),
            reversal_require_structure_confirmation=not bool(args.disable_reversal_structure_confirmation),
        )
        grid: dict[str, list[object]] = {
            "risk_reward": _parse_float_list(args.risk_rewards),
            "max_holding_bars": _parse_int_list(args.max_holding_bars_list),
        }
        if str(args.trend_min_scores).strip():
            grid["trend_min_score"] = _parse_float_list(args.trend_min_scores)
        grid.update(_parse_generic_sweep_grid(args.grid, config))
        result = run_single_strategy_parameter_sweep(config, grid=grid, save=bool(args.output_dir))
        print(result.table.to_string(index=False))
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
            _print_saved_artifact_manifest(output_dir)
            print(f"sweep.csv saved: {output_dir / 'sweep.csv'}")
            print(f"pareto.csv saved: {output_dir / 'pareto.csv'}")
            print(f"parameter_summary.csv saved: {output_dir / 'parameter_summary.csv'}")
            print(f"case_setup_order_decision_stats.csv saved: {output_dir / 'case_setup_order_decision_stats.csv'}")
            print(f"case_setup_strategy_filter_stats.csv saved: {output_dir / 'case_setup_strategy_filter_stats.csv'}")
            print(f"summary.json saved: {output_dir / 'summary.json'}")
            print(f"case_configs.jsonl saved: {output_dir / 'case_configs.jsonl'}")
        return

    if args.command == "portfolio-backtest":
        config = PortfolioExperimentConfig(
            name="portfolio-backtest",
            data_root=str(args.data_root),
            symbols=symbols,
            timeframe=args.timeframe,
            higher_timeframe=str(args.higher_timeframe),
            higher_timeframe_max_age_minutes=args.higher_timeframe_max_age_minutes,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            detectors=tuple(item.strip() for item in args.detectors.split(",") if item.strip()),
            risk_reward=float(args.risk_reward),
            max_holding_bars=int(args.max_holding_bars),
            max_actual_risk_pct=args.max_actual_risk_pct,
            max_chase_pct=args.max_chase_pct,
            side_mode=str(args.side_mode),
            min_coverage_ratio=args.min_coverage_ratio,
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            initial_equity=float(args.initial_equity),
            max_open_positions=int(args.max_open_positions),
            capital_per_trade=args.capital_per_trade,
            risk_per_trade=args.risk_per_trade,
            max_capital_per_trade=float(args.max_capital_per_trade),
            short_margin_rate=float(args.short_margin_rate),
            reserve_cash=float(args.reserve_cash),
            allow_same_symbol_overlap=bool(args.allow_same_symbol_overlap),
            strategy_priority=_parse_int_mapping(args.strategy_priority),
            strategy_capital_limit=_parse_float_mapping(args.strategy_capital_limit),
            sector_capital_limit=_parse_float_mapping(args.sector_capital_limit),
            symbol_sector_map=_parse_text_mapping(args.symbol_sector_map),
            sector_metadata_key=str(args.sector_metadata_key),
            default_sector=str(args.default_sector),
            intrabar_exit_policy=str(args.intrabar_exit_policy),
            trailing_take_profit_activation_pct=float(args.trailing_take_profit_activation_pct),
            trailing_take_profit_drawdown_pct=float(args.trailing_take_profit_drawdown_pct),
            trailing_take_profit_ma_period=int(args.trailing_take_profit_ma_period),
            **_terminal_false_breakout_kwargs(args),
            trend_lookback=int(args.trend_lookback),
            trend_min_score=float(args.trend_min_score),
            trend_strong_close_pos=float(args.trend_strong_close_pos),
            trend_min_body_ratio=float(args.trend_min_body_ratio),
            trend_pullback_lookback=int(args.trend_pullback_lookback),
            trend_h2_min_pullback_legs=int(args.trend_h2_min_pullback_legs),
            range_lookback=int(args.range_lookback),
            range_middle_low=float(args.range_middle_low),
            range_middle_high=float(args.range_middle_high),
            range_false_break_buffer=float(args.range_false_break_buffer),
            range_strong_close_pos=float(args.range_strong_close_pos),
            range_min_score=float(args.range_min_score),
            channel_method=str(args.channel_method),
            channel_lookback=int(args.channel_lookback),
            channel_sigma_multiple=float(args.channel_sigma_multiple),
            channel_break_buffer=float(args.channel_break_buffer),
            channel_swing_left_bars=int(args.channel_swing_left_bars),
            channel_swing_right_bars=int(args.channel_swing_right_bars),
            reversal_lookback=int(args.reversal_lookback),
            reversal_strong_close_pos=float(args.reversal_strong_close_pos),
            reversal_min_body_ratio=float(args.reversal_min_body_ratio),
            reversal_old_extreme_tolerance_pct=float(args.reversal_old_extreme_tolerance_pct),
            reversal_require_old_extreme_test=not bool(args.disable_reversal_old_extreme_test),
            reversal_require_structure_confirmation=not bool(args.disable_reversal_structure_confirmation),
            strict_data_quality=not bool(args.allow_bad_data),
            output_dir=args.output_dir,
        )
        experiment = run_portfolio_experiment(config, save=bool(args.output_dir))
        result = experiment.backtest
        print(result.stats)
        if not result.equity_curve.empty:
            print(result.equity_curve.tail(20).to_string(index=False))
        if not result.trades.empty:
            print(result.trades.to_string(index=False))
        if args.benchmark:
            report = build_portfolio_benchmark_report(experiment)
            if args.output_dir:
                save_portfolio_benchmark(config, report)
            print(
                {
                    "bar_count": report.bar_count,
                    "elapsed_seconds": report.elapsed_seconds,
                    "bars_per_second": report.bars_per_second,
                }
            )
        if args.output_dir:
            _print_saved_artifact_manifest(args.output_dir)
        return

    if args.command == "portfolio-sweep":
        config = PortfolioExperimentConfig(
            name="portfolio-sweep",
            data_root=str(args.data_root),
            symbols=symbols,
            timeframe=args.timeframe,
            higher_timeframe=str(args.higher_timeframe),
            higher_timeframe_max_age_minutes=args.higher_timeframe_max_age_minutes,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
            detectors=tuple(item.strip() for item in args.detectors.split(",") if item.strip()),
            max_actual_risk_pct=args.max_actual_risk_pct,
            max_chase_pct=args.max_chase_pct,
            side_mode=str(args.side_mode),
            min_coverage_ratio=args.min_coverage_ratio,
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
            initial_equity=float(args.initial_equity),
            capital_per_trade=args.capital_per_trade,
            risk_per_trade=args.risk_per_trade,
            max_capital_per_trade=float(args.max_capital_per_trade),
            short_margin_rate=float(args.short_margin_rate),
            reserve_cash=float(args.reserve_cash),
            allow_same_symbol_overlap=bool(args.allow_same_symbol_overlap),
            strategy_priority=_parse_int_mapping(args.strategy_priority),
            strategy_capital_limit=_parse_float_mapping(args.strategy_capital_limit),
            sector_capital_limit=_parse_float_mapping(args.sector_capital_limit),
            symbol_sector_map=_parse_text_mapping(args.symbol_sector_map),
            sector_metadata_key=str(args.sector_metadata_key),
            default_sector=str(args.default_sector),
            intrabar_exit_policy=str(args.intrabar_exit_policy),
            trailing_take_profit_activation_pct=float(args.trailing_take_profit_activation_pct),
            trailing_take_profit_drawdown_pct=float(args.trailing_take_profit_drawdown_pct),
            trailing_take_profit_ma_period=int(args.trailing_take_profit_ma_period),
            **_terminal_false_breakout_kwargs(args),
            trend_lookback=int(args.trend_lookback),
            trend_min_score=float(args.trend_min_score),
            trend_strong_close_pos=float(args.trend_strong_close_pos),
            trend_min_body_ratio=float(args.trend_min_body_ratio),
            trend_pullback_lookback=int(args.trend_pullback_lookback),
            trend_h2_min_pullback_legs=int(args.trend_h2_min_pullback_legs),
            range_lookback=int(args.range_lookback),
            range_middle_low=float(args.range_middle_low),
            range_middle_high=float(args.range_middle_high),
            range_false_break_buffer=float(args.range_false_break_buffer),
            range_strong_close_pos=float(args.range_strong_close_pos),
            range_min_score=float(args.range_min_score),
            channel_method=str(args.channel_method),
            channel_lookback=int(args.channel_lookback),
            channel_sigma_multiple=float(args.channel_sigma_multiple),
            channel_break_buffer=float(args.channel_break_buffer),
            channel_swing_left_bars=int(args.channel_swing_left_bars),
            channel_swing_right_bars=int(args.channel_swing_right_bars),
            reversal_lookback=int(args.reversal_lookback),
            reversal_strong_close_pos=float(args.reversal_strong_close_pos),
            reversal_min_body_ratio=float(args.reversal_min_body_ratio),
            reversal_old_extreme_tolerance_pct=float(args.reversal_old_extreme_tolerance_pct),
            reversal_require_old_extreme_test=not bool(args.disable_reversal_old_extreme_test),
            reversal_require_structure_confirmation=not bool(args.disable_reversal_structure_confirmation),
            strict_data_quality=not bool(args.allow_bad_data),
            output_dir=args.output_dir,
        )
        result = run_portfolio_parameter_sweep(
            config,
            grid={
                "risk_reward": _parse_float_list(args.risk_rewards),
                "max_holding_bars": _parse_int_list(args.max_holding_bars_list),
                "max_open_positions": _parse_int_list(args.max_open_positions_list),
                **_parse_generic_sweep_grid(args.grid, config),
            },
            save=bool(args.output_dir),
        )
        print(result.table.to_string(index=False))
        if args.output_dir:
            output_dir = Path(args.output_dir).expanduser()
            _print_saved_artifact_manifest(output_dir)
            print(f"sweep.csv saved: {output_dir / 'sweep.csv'}")
            print(f"pareto.csv saved: {output_dir / 'pareto.csv'}")
            print(f"parameter_summary.csv saved: {output_dir / 'parameter_summary.csv'}")
            print(f"case_setup_order_decision_stats.csv saved: {output_dir / 'case_setup_order_decision_stats.csv'}")
            print(f"case_setup_strategy_filter_stats.csv saved: {output_dir / 'case_setup_strategy_filter_stats.csv'}")
            print(f"summary.json saved: {output_dir / 'summary.json'}")
            print(f"case_configs.jsonl saved: {output_dir / 'case_configs.jsonl'}")
        return

    bundle = repo.load_backtest_data(
        timeframe=args.timeframe,
        symbols=symbols,
        start=args.start,
        end=args.end,
    )
    scanned = scan_bars(bundle.bars, StrategyConfig())
    result = run_backtest(scanned, BacktestConfig())
    print(result.stats)
    if not result.trades.empty:
        print(result.trades.to_string(index=False))


def _add_tdx_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime",
        choices=["auto", "local", "parallels"],
        default="auto",
        help="TDX 执行位置；Mac 默认调度到 Parallels，Windows 默认本地执行。",
    )
    parser.add_argument("--parallels-vm", default="", help="Parallels 虚拟机名称，默认 Windows 11。")
    parser.add_argument(
        "--windows-python",
        default="",
        help=(
            "Windows 内 Python 可执行文件，默认 "
            r"C:\Users\Public\venvs\trending-winning\Scripts\python.exe。"
        ),
    )
    parser.add_argument("--windows-repo", default="", help="Windows 内本仓库路径，默认由 Mac 共享目录推导。")


def _resolve_tdx_runtime(runtime: str) -> str:
    if runtime == "auto":
        return "parallels" if sys.platform == "darwin" else "local"
    return runtime


def _run_tdx_cli_in_parallels(args: argparse.Namespace) -> None:
    config = _parallels_config_from_args(args)
    result = run_parallels_tdx_command(config=config, cli_args=_tdx_forward_args(args))
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _parallels_config_from_args(args: argparse.Namespace) -> ParallelsTdxConfig:
    default = default_parallels_tdx_config(cwd=Path.cwd())
    return ParallelsTdxConfig(
        vm_name=args.parallels_vm or default.vm_name,
        windows_python=args.windows_python or default.windows_python,
        windows_repo=args.windows_repo or default.windows_repo,
    )


def _tdx_forward_args(args: argparse.Namespace) -> list[str]:
    if args.command == "tdx-doctor":
        return [
            "tdx-doctor",
            "--runtime",
            "local",
            "--symbols",
            args.symbols,
            "--timeframes",
            args.timeframes,
            "--start",
            args.start,
            "--end",
            args.end,
            "--adjust",
            args.adjust,
            "--tdx-path",
            mac_path_to_parallels_shared_path(args.tdx_path),
        ]
    if args.command == "fetch":
        return [
            "fetch",
            "--runtime",
            "local",
            "--symbols",
            args.symbols,
            "--timeframe",
            args.timeframe,
            "--start",
            args.start,
            "--end",
            args.end,
            "--adjust",
            args.adjust,
            "--data-root",
            mac_path_to_parallels_shared_path(args.data_root),
            "--tdx-path",
            mac_path_to_parallels_shared_path(args.tdx_path),
        ]
    if args.command == "prepare-data":
        forwarded = [
            "prepare-data",
            "--runtime",
            "local",
            "--symbols",
            args.symbols,
            "--timeframes",
            args.timeframes,
            "--start",
            args.start,
            "--end",
            args.end,
            "--adjust",
            args.adjust,
            "--data-root",
            mac_path_to_parallels_shared_path(args.data_root),
            "--tdx-path",
            mac_path_to_parallels_shared_path(args.tdx_path),
        ]
        if args.min_coverage_ratio is not None:
            forwarded.extend(["--min-coverage-ratio", str(args.min_coverage_ratio)])
        if args.allow_incomplete_after_update:
            forwarded.append("--allow-incomplete-after-update")
        return forwarded
    raise ValueError(f"不支持通过 Parallels 运行的命令：{args.command}")


if __name__ == "__main__":
    main()
