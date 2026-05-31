from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import pandas as pd

from trending_winning.backtest.experiment_models import (
    PortfolioExperimentConfig,
    PortfolioSweepResult,
    SingleStrategyExperimentConfig,
    SingleStrategySweepResult,
)

DATA_SCOPE_SWEEP_FIELDS = {
    "data_root",
    "symbols",
    "timeframe",
    "higher_timeframe",
    "start",
    "end",
    "adjust",
    "strict_data_quality",
    "min_coverage_ratio",
}

DETECTOR_PARAMETER_FIELDS = {
    "trend": frozenset(
        {
            "trend_lookback",
            "trend_min_score",
            "trend_strong_close_pos",
            "trend_min_body_ratio",
            "trend_pullback_lookback",
            "trend_h2_min_pullback_legs",
        }
    ),
    "range": frozenset(
        {
            "range_lookback",
            "range_middle_low",
            "range_middle_high",
            "range_false_break_buffer",
            "range_strong_close_pos",
            "range_min_score",
        }
    ),
    "channel": frozenset(
        {
            "channel_method",
            "channel_lookback",
            "channel_sigma_multiple",
            "channel_break_buffer",
            "channel_swing_left_bars",
            "channel_swing_right_bars",
        }
    ),
    "reversal": frozenset(
        {
            "reversal_lookback",
            "reversal_strong_close_pos",
            "reversal_min_body_ratio",
            "reversal_old_extreme_tolerance_pct",
            "reversal_require_old_extreme_test",
            "reversal_require_structure_confirmation",
        }
    ),
}

ALL_DETECTOR_PARAMETER_FIELDS = frozenset().union(*DETECTOR_PARAMETER_FIELDS.values())
NON_REPRODUCIBLE_CONFIG_HASH_FIELDS = frozenset({"name", "data_root", "output_dir"})


def sweep_variants(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    grid: Mapping[str, Sequence[object]],
) -> list[PortfolioExperimentConfig | SingleStrategyExperimentConfig]:
    if not grid:
        raise ValueError("grid 不能为空。")
    config_fields = {field.name for field in fields(type(config))}
    unknown = set(grid).difference(config_fields)
    if unknown:
        raise ValueError(f"grid 包含不支持的配置字段：{', '.join(sorted(unknown))}")
    data_scope_fields = set(grid).intersection(DATA_SCOPE_SWEEP_FIELDS)
    if data_scope_fields:
        raise ValueError(f"不能在同一次 sweep 中改变数据范围字段：{', '.join(sorted(data_scope_fields))}")
    raw_keys = list(grid)
    raw_value_lists = [list(grid[key]) for key in raw_keys]
    empty_keys = [key for key, values in zip(raw_keys, raw_value_lists, strict=False) if not values]
    if empty_keys:
        raise ValueError(f"grid 字段不能为空：{', '.join(empty_keys)}")
    effective_grid = effective_sweep_grid(config, grid)
    if not effective_grid:
        return [config]
    keys = list(effective_grid)
    raw_value_lists = [list(effective_grid[key]) for key in keys]
    value_lists = [_deduplicate_sweep_grid_values(values) for values in raw_value_lists]
    variants = [replace(config, **dict(zip(keys, values, strict=False))) for values in product(*value_lists)]
    return _deduplicate_sweep_variants(variants)


def effective_sweep_grid(
    config: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    grid: Mapping[str, Sequence[object]],
) -> dict[str, list[object]]:
    """过滤单策略无效 detector 参数，避免未启用模块进入 sweep 热路径。"""
    normalized = {str(key): list(values) for key, values in grid.items()}
    if isinstance(config, PortfolioExperimentConfig) and "detectors" in normalized:
        return normalized
    if isinstance(config, SingleStrategyExperimentConfig) and "detector" in normalized:
        return normalized

    active_fields = active_detector_parameter_fields(config)
    return {
        key: values
        for key, values in normalized.items()
        if key not in ALL_DETECTOR_PARAMETER_FIELDS or key in active_fields
    }


def sweep_parameter_record(
    base: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    variant: PortfolioExperimentConfig | SingleStrategyExperimentConfig,
    keys: Sequence[str],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for key in keys:
        value = getattr(variant, key)
        record[key] = ",".join(value) if isinstance(value, tuple) else value
    for key in ("detectors", "detector", "side_mode", "intrabar_exit_policy"):
        if key not in record:
            if not hasattr(base, key):
                continue
            value = getattr(base, key)
            record[key] = ",".join(value) if isinstance(value, tuple) else value
    return record


def case_config_hash(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> str:
    """给完整实验配置生成稳定指纹，用于 sweep 行跨机器复现和对照。"""
    return _stable_case_config_hash(config)


def _stable_case_config_hash(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> str:
    payload = json.dumps(case_config_hash_payload(config), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def case_config_hash_payload(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> dict[str, object]:
    payload = json_ready(asdict(config))
    for key in NON_REPRODUCIBLE_CONFIG_HASH_FIELDS:
        payload.pop(key, None)
    active_fields = active_detector_parameter_fields(config)
    for key in ALL_DETECTOR_PARAMETER_FIELDS.difference(active_fields):
        payload.pop(key, None)
    return payload


def active_detector_parameter_fields(config: PortfolioExperimentConfig | SingleStrategyExperimentConfig) -> set[str]:
    detector_names = config.detectors if isinstance(config, PortfolioExperimentConfig) else (config.detector,)
    active_fields: set[str] = set()
    for detector_name in detector_names:
        active_fields.update(DETECTOR_PARAMETER_FIELDS.get(detector_name, frozenset()))
    if str(config.higher_timeframe).strip():
        active_fields.update(DETECTOR_PARAMETER_FIELDS["trend"])
    return active_fields


def sweep_case_config_records(result: PortfolioSweepResult | SingleStrategySweepResult) -> list[dict[str, object]]:
    """按 sweep 表排序输出每个 case 的完整配置，便于从结果行直接复现实验。"""
    records_by_hash: dict[str, dict[str, object]] = {}
    for case_index, variant in enumerate(sweep_variants(result.config, result.grid), start=1):
        config_hash = case_config_hash(variant)
        records_by_hash[config_hash] = {
            "case_name": f"{result.config.name}-{case_index:03d}",
            "case_config_hash": config_hash,
            "grid_fields": list(result.grid),
            "config": json_ready(asdict(variant)),
        }
    if result.table.empty or "case_config_hash" not in result.table.columns:
        return list(records_by_hash.values())

    records: list[dict[str, object]] = []
    for row in result.table.to_dict("records"):
        config_hash = str(row["case_config_hash"])
        record = records_by_hash.get(config_hash)
        if record is None:
            raise ValueError(f"sweep 表包含未知 case_config_hash：{config_hash}")
        enriched = dict(record)
        for column in ("sweep_rank", "pareto_rank", "is_pareto_efficient"):
            if column in row:
                enriched[column] = row[column]
        records.append(enriched)
    return records


def load_sweep_case_config(
    path: str | Path,
    *,
    case_config_hash: str = "",
    case_name: str = "",
) -> PortfolioExperimentConfig | SingleStrategyExperimentConfig:
    """从 case_configs.jsonl 读取单个 case 的完整配置，用于精确回放参数遍历结果。"""
    if not case_config_hash and not case_name:
        raise ValueError("必须提供 case_config_hash 或 case_name。")
    records = read_jsonl(Path(path).expanduser())
    matches = [
        record
        for record in records
        if (not case_config_hash or str(record.get("case_config_hash", "")) == case_config_hash)
        and (not case_name or str(record.get("case_name", "")) == case_name)
    ]
    if not matches:
        raise ValueError("未找到匹配的 sweep case 配置。")
    if len(matches) > 1:
        raise ValueError("匹配到多个 sweep case 配置，请同时指定 case_config_hash 和 case_name。")
    config_payload = matches[0].get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("case 配置缺少 config 对象。")
    config = experiment_config_from_payload(config_payload)
    recorded_hash = str(matches[0].get("case_config_hash", ""))
    actual_hash = _stable_case_config_hash(config)
    if recorded_hash and recorded_hash != actual_hash:
        raise ValueError("case_config_hash 与 config 内容不一致，拒绝回放被篡改或损坏的 case 配置。")
    return config


def experiment_config_from_payload(
    payload: Mapping[str, object],
) -> PortfolioExperimentConfig | SingleStrategyExperimentConfig:
    data = dict(payload)
    if "symbols" in data:
        data["symbols"] = tuple(data["symbols"]) if isinstance(data["symbols"], list) else data["symbols"]
    if "detectors" in data:
        data["detectors"] = tuple(data["detectors"]) if isinstance(data["detectors"], list) else data["detectors"]
        return PortfolioExperimentConfig(**data)
    if "detector" in data:
        return SingleStrategyExperimentConfig(**data)
    raise ValueError("case 配置无法识别为单策略或组合实验配置。")


def json_ready(value):
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def json_dump(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    lines = [
        json.dumps(json_ready(record), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))
        for record in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"case 配置文件不存在：{path}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"case 配置第 {line_number} 行不是 JSON 对象。")
        records.append(payload)
    return records


def _deduplicate_sweep_grid_values(values: Sequence[object]) -> list[object]:
    """在笛卡尔积展开前去掉重复参数值，避免重复配置进入热路径。"""
    seen: set[str] = set()
    deduplicated: list[object] = []
    for value in values:
        fingerprint = _sweep_grid_value_fingerprint(value)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduplicated.append(value)
    return deduplicated


def _sweep_grid_value_fingerprint(value: object) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _deduplicate_sweep_variants(
    variants: Sequence[PortfolioExperimentConfig | SingleStrategyExperimentConfig],
) -> list[PortfolioExperimentConfig | SingleStrategyExperimentConfig]:
    """按完整配置指纹去掉重复 case，避免重复 grid 值造成无效回测。"""
    seen: set[str] = set()
    deduplicated: list[PortfolioExperimentConfig | SingleStrategyExperimentConfig] = []
    for variant in variants:
        config_hash = case_config_hash(variant)
        if config_hash in seen:
            continue
        seen.add(config_hash)
        deduplicated.append(variant)
    return deduplicated

