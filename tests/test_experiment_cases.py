from __future__ import annotations

from dataclasses import asdict, replace
import json
import sys

import pytest

from trending_winning.backtest.experiment_models import (
    PortfolioExperimentConfig,
    PortfolioSweepResult,
    SingleStrategyExperimentConfig,
)


def test_experiment_cases_import_without_experiment_runner(tmp_path) -> None:
    sys.modules.pop("trending_winning.backtest.experiment", None)

    from trending_winning.backtest.experiment_cases import (
        case_config_hash,
        load_sweep_case_config,
        sweep_case_config_records,
        sweep_parameter_record,
        sweep_variants,
        write_jsonl,
    )

    base = SingleStrategyExperimentConfig(
        name="base",
        data_root=str(tmp_path / "mac"),
        output_dir=str(tmp_path / "runs"),
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    same_reproducible_config = replace(base, name="renamed", data_root=str(tmp_path / "windows"), output_dir="")
    changed_active_param = replace(base, trend_min_score=0.5)
    changed_inactive_param = replace(base, range_min_score=0.1)

    variants = sweep_variants(base, {"risk_reward": [2.0, 2.0, 1.5], "range_min_score": [0.1, 0.9]})
    result = PortfolioSweepResult(
        config=PortfolioExperimentConfig(
            name="portfolio",
            data_root="/data",
            symbols=("000001.SZ",),
            timeframe="30m",
            start="2026-05-25",
            end="2026-05-25",
        ),
        grid={"risk_reward": [2.0]},
        table=_case_table("portfolio-001", case_config_hash(PortfolioExperimentConfig(
            name="portfolio",
            data_root="/data",
            symbols=("000001.SZ",),
            timeframe="30m",
            start="2026-05-25",
            end="2026-05-25",
        ))),
        data_coverage=_empty_frame(),
        input_bar_count=0,
        filtered_limit_open_count=0,
        elapsed_seconds=0.1,
    )
    case_file = tmp_path / "case_configs.jsonl"
    write_jsonl(case_file, sweep_case_config_records(result))
    loaded = load_sweep_case_config(case_file, case_name="portfolio-001")

    assert "trending_winning.backtest.experiment" not in sys.modules
    assert [variant.risk_reward for variant in variants] == [2.0, 1.5]
    assert sweep_parameter_record(base, replace(base, risk_reward=1.5), ["risk_reward"])["risk_reward"] == 1.5
    assert case_config_hash(base) == case_config_hash(same_reproducible_config)
    assert case_config_hash(base) != case_config_hash(changed_active_param)
    assert case_config_hash(base) == case_config_hash(changed_inactive_param)
    assert loaded.name == "portfolio"


def test_load_sweep_case_config_rejects_tampered_hash(tmp_path) -> None:
    from trending_winning.backtest.experiment_cases import case_config_hash, load_sweep_case_config

    config = SingleStrategyExperimentConfig(
        name="single",
        data_root="/data",
        symbols=("000001.SZ",),
        timeframe="30m",
        start="2026-05-25",
        end="2026-05-25",
        detector="trend",
    )
    payload = {
        "case_name": "single-001",
        "case_config_hash": case_config_hash(config),
        "config": {**asdict(config), "risk_reward": 1.5},
    }
    case_file = tmp_path / "case_configs.jsonl"
    case_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n")

    with pytest.raises(ValueError, match="case_config_hash 与 config 内容不一致"):
        load_sweep_case_config(case_file, case_name="single-001")


def _case_table(case_name: str, case_hash: str):
    import pandas as pd

    return pd.DataFrame(
        {
            "case_name": [case_name],
            "case_config_hash": [case_hash],
            "sweep_rank": [1],
            "pareto_rank": [1],
            "is_pareto_efficient": [True],
        }
    )


def _empty_frame():
    import pandas as pd

    return pd.DataFrame()
