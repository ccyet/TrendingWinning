from __future__ import annotations

from inspect import getsource

import pytest

import trending_winning.strategies.suite as suite_module
from trending_winning.strategies.suite import StrategySuiteConfig, create_default_strategy_suite, create_strategy_for_detector


def test_default_strategy_suite_builds_independent_detector_strategies() -> None:
    suite = create_default_strategy_suite(
        StrategySuiteConfig(
            enabled=("trend", "range", "channel", "reversal"),
            risk_reward=1.6,
            max_holding_bars=9,
            max_actual_risk_pct=0.03,
            max_chase_pct=0.02,
            trend_lookback=5,
            trend_strong_close_pos=0.7,
            trend_min_body_ratio=0.55,
            trend_pullback_lookback=6,
            trend_h2_min_pullback_legs=3,
            range_lookback=6,
            range_middle_low=0.2,
            range_middle_high=0.8,
            range_false_break_buffer=0.01,
            range_strong_close_pos=0.7,
            range_min_score=0.9,
            channel_lookback=7,
            channel_method="swing",
            channel_break_buffer=0.02,
            channel_swing_left_bars=3,
            channel_swing_right_bars=4,
            reversal_old_extreme_tolerance_pct=0.02,
            reversal_strong_close_pos=0.7,
            reversal_min_body_ratio=0.5,
            reversal_require_old_extreme_test=False,
            reversal_require_structure_confirmation=False,
        )
    )

    assert [strategy.name for strategy in suite] == [
        "trend_signal_bar",
        "range_signal_bar",
        "channel_signal_bar",
        "reversal_signal_bar",
    ]
    assert [strategy.detector.name for strategy in suite] == ["trend", "range", "channel", "reversal"]
    assert [strategy.config.risk_reward for strategy in suite] == [1.6, 1.6, 1.6, 1.6]
    assert [strategy.config.max_holding_bars for strategy in suite] == [9, 9, 9, 9]
    assert [strategy.config.max_actual_risk_pct for strategy in suite] == [0.03, 0.03, 0.03, 0.03]
    assert [strategy.config.max_chase_pct for strategy in suite] == [0.02, 0.02, 0.02, 0.02]
    assert suite[0].detector.config.h2_min_pullback_legs == 3
    assert suite[0].detector.config.strong_close_pos == 0.7
    assert suite[0].detector.config.min_body_ratio == 0.55
    assert suite[0].detector.config.pullback_lookback == 6
    assert suite[1].detector.config.middle_low == 0.2
    assert suite[1].detector.config.middle_high == 0.8
    assert suite[1].detector.config.false_break_buffer == 0.01
    assert suite[1].detector.config.strong_close_pos == 0.7
    assert suite[1].detector.config.min_range_score == 0.9
    assert suite[2].detector.config.channel_method == "swing"
    assert suite[2].detector.config.break_buffer == 0.02
    assert suite[2].detector.config.swing_left_bars == 3
    assert suite[2].detector.config.swing_right_bars == 4
    assert suite[3].detector.config.old_extreme_tolerance_pct == 0.02
    assert suite[3].detector.config.strong_close_pos == 0.7
    assert suite[3].detector.config.min_body_ratio == 0.5
    assert suite[3].detector.config.require_old_extreme_test is False
    assert suite[3].detector.config.require_structure_confirmation is False
    assert len({id(strategy.detector) for strategy in suite}) == 4


def test_create_strategy_for_detector_builds_one_detector_strategy() -> None:
    strategy = create_strategy_for_detector(
        "range",
        StrategySuiteConfig(
            enabled=("trend", "range", "channel"),
            risk_reward=1.4,
            max_holding_bars=7,
            max_actual_risk_pct=0.04,
            max_chase_pct=0.03,
            range_lookback=9,
            range_middle_low=0.2,
            range_middle_high=0.8,
            range_min_score=0.7,
        ),
    )

    assert strategy.name == "range_signal_bar"
    assert strategy.detector.name == "range"
    assert strategy.detector.config.lookback == 9
    assert strategy.detector.config.middle_low == 0.2
    assert strategy.detector.config.middle_high == 0.8
    assert strategy.detector.config.min_range_score == 0.7
    assert strategy.config.risk_reward == 1.4
    assert strategy.config.max_holding_bars == 7
    assert strategy.config.max_actual_risk_pct == 0.04
    assert strategy.config.max_chase_pct == 0.03


def test_create_strategy_for_detector_ignores_suite_enabled_list() -> None:
    strategy = create_strategy_for_detector("trend", StrategySuiteConfig(enabled=(), trend_lookback=6))

    assert strategy.name == "trend_signal_bar"
    assert strategy.detector.name == "trend"
    assert strategy.detector.config.lookback == 6


def test_single_detector_strategy_ignores_disabled_detector_parameters() -> None:
    cfg = StrategySuiteConfig(
        enabled=("trend",),
        trend_lookback=6,
        range_middle_low=0.9,
        range_middle_high=0.1,
        channel_method="bad-method",
        reversal_strong_close_pos=2.0,
    )

    strategy = create_strategy_for_detector("trend", cfg)
    suite = create_default_strategy_suite(cfg)

    assert strategy.detector.name == "trend"
    assert strategy.detector.config.lookback == 6
    assert [item.detector.name for item in suite] == ["trend"]


def test_strategy_suite_does_not_import_all_concrete_detectors_at_module_load() -> None:
    source = getsource(suite_module).split("SUPPORTED_STRATEGY_DETECTORS", maxsplit=1)[0]

    assert "trending_winning.detectors.trend" not in source
    assert "trending_winning.detectors.range" not in source
    assert "trending_winning.detectors.channel" not in source
    assert "trending_winning.detectors.reversal" not in source


def test_strategy_suite_rejects_empty_detector_list() -> None:
    with pytest.raises(ValueError, match="至少需要启用一个 detector"):
        create_default_strategy_suite(StrategySuiteConfig(enabled=()))


def test_strategy_suite_rejects_duplicate_detectors() -> None:
    with pytest.raises(ValueError, match="detector 不能重复"):
        create_default_strategy_suite(StrategySuiteConfig(enabled=("trend", "trend")))
