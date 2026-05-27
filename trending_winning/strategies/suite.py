from __future__ import annotations

from dataclasses import dataclass

from trending_winning.strategies.signal_bar import SignalBarStopStrategy, SignalBarStopStrategyConfig

SUPPORTED_STRATEGY_DETECTORS = ("trend", "range", "channel", "reversal")


@dataclass(frozen=True)
class StrategySuiteConfig:
    """默认组合策略参数；每个策略绑定一个独立 detector。"""

    enabled: tuple[str, ...] = ("trend", "range", "channel")
    risk_reward: float = 2.0
    max_holding_bars: int = 12
    max_actual_risk_pct: float | None = None
    max_chase_pct: float | None = None
    trend_lookback: int = 20
    trend_min_score: float = 1.0
    trend_strong_close_pos: float = 0.65
    trend_min_body_ratio: float = 0.45
    trend_pullback_lookback: int = 5
    trend_h2_min_pullback_legs: int = 2
    range_lookback: int = 20
    range_middle_low: float = 0.25
    range_middle_high: float = 0.75
    range_false_break_buffer: float = 0.0
    range_strong_close_pos: float = 0.65
    range_min_score: float = 0.8
    channel_method: str = "regression"
    channel_lookback: int = 40
    channel_sigma_multiple: float = 2.0
    channel_break_buffer: float = 0.0
    channel_swing_left_bars: int = 2
    channel_swing_right_bars: int = 2
    reversal_lookback: int = 20
    reversal_strong_close_pos: float = 0.65
    reversal_min_body_ratio: float = 0.45
    reversal_old_extreme_tolerance_pct: float = 0.01
    reversal_require_old_extreme_test: bool = True
    reversal_require_structure_confirmation: bool = True


def create_default_strategy_suite(config: StrategySuiteConfig | None = None) -> list[SignalBarStopStrategy]:
    cfg = config or StrategySuiteConfig()
    _validate_suite_config(cfg)

    return [create_strategy_for_detector(detector_name, cfg) for detector_name in cfg.enabled]


def create_strategy_for_detector(
    detector_name: str,
    config: StrategySuiteConfig | None = None,
) -> SignalBarStopStrategy:
    """只为一个 detector 构造策略；单策略回测不需要创建组合策略套件。"""
    cfg = config or StrategySuiteConfig(enabled=(detector_name,))
    _validate_detector_name(detector_name)
    _validate_strategy_parameters(cfg, enabled_detectors=(detector_name,))
    return SignalBarStopStrategy(
        detector=_create_detector(detector_name, cfg),
        config=SignalBarStopStrategyConfig(
            name=f"{detector_name}_signal_bar",
            risk_reward=cfg.risk_reward,
            max_holding_bars=cfg.max_holding_bars,
            max_actual_risk_pct=cfg.max_actual_risk_pct,
            max_chase_pct=cfg.max_chase_pct,
        ),
    )


def _create_detector(name: str, cfg: StrategySuiteConfig):
    if name == "trend":
        from trending_winning.detectors.trend import TrendDetector, TrendDetectorConfig

        return TrendDetector(
            TrendDetectorConfig(
                lookback=cfg.trend_lookback,
                min_trend_score=cfg.trend_min_score,
                strong_close_pos=cfg.trend_strong_close_pos,
                min_body_ratio=cfg.trend_min_body_ratio,
                pullback_lookback=cfg.trend_pullback_lookback,
                h2_min_pullback_legs=cfg.trend_h2_min_pullback_legs,
            )
        )
    if name == "range":
        from trending_winning.detectors.range import RangeDetector, RangeDetectorConfig

        return RangeDetector(
            RangeDetectorConfig(
                lookback=cfg.range_lookback,
                middle_low=cfg.range_middle_low,
                middle_high=cfg.range_middle_high,
                false_break_buffer=cfg.range_false_break_buffer,
                strong_close_pos=cfg.range_strong_close_pos,
                min_range_score=cfg.range_min_score,
            )
        )
    if name == "channel":
        from trending_winning.detectors.channel import ChannelDetector, ChannelDetectorConfig

        return ChannelDetector(
            ChannelDetectorConfig(
                lookback=cfg.channel_lookback,
                sigma_multiple=cfg.channel_sigma_multiple,
                break_buffer=cfg.channel_break_buffer,
                channel_method=cfg.channel_method,
                swing_left_bars=cfg.channel_swing_left_bars,
                swing_right_bars=cfg.channel_swing_right_bars,
            )
        )
    if name == "reversal":
        from trending_winning.detectors.reversal import ReversalDetector, ReversalDetectorConfig

        return ReversalDetector(
            ReversalDetectorConfig(
                lookback=cfg.reversal_lookback,
                strong_close_pos=cfg.reversal_strong_close_pos,
                min_body_ratio=cfg.reversal_min_body_ratio,
                old_extreme_tolerance_pct=cfg.reversal_old_extreme_tolerance_pct,
                require_old_extreme_test=cfg.reversal_require_old_extreme_test,
                require_structure_confirmation=cfg.reversal_require_structure_confirmation,
            )
        )
    raise ValueError(f"不支持的 detector：{name}")


def _validate_suite_config(cfg: StrategySuiteConfig) -> None:
    _validate_enabled_detectors(cfg.enabled)
    _validate_strategy_parameters(cfg, enabled_detectors=cfg.enabled)


def _validate_enabled_detectors(enabled: tuple[str, ...]) -> None:
    if not enabled:
        raise ValueError("至少需要启用一个 detector。")
    if len(set(enabled)) != len(enabled):
        raise ValueError("detector 不能重复。")
    unknown = set(enabled).difference(SUPPORTED_STRATEGY_DETECTORS)
    if unknown:
        raise ValueError(f"不支持的 detector：{', '.join(sorted(unknown))}")


def _validate_detector_name(detector_name: str) -> None:
    if detector_name not in SUPPORTED_STRATEGY_DETECTORS:
        raise ValueError(f"不支持的 detector：{detector_name}")


def _validate_strategy_parameters(cfg: StrategySuiteConfig, *, enabled_detectors: tuple[str, ...]) -> None:
    _validate_common_strategy_parameters(cfg)
    for detector_name in enabled_detectors:
        _validate_detector_parameters(detector_name, cfg)


def _validate_common_strategy_parameters(cfg: StrategySuiteConfig) -> None:
    if cfg.risk_reward <= 0:
        raise ValueError("risk_reward 必须大于 0。")
    if cfg.max_holding_bars < 1:
        raise ValueError("max_holding_bars 至少需要 1。")
    if cfg.max_actual_risk_pct is not None and cfg.max_actual_risk_pct <= 0:
        raise ValueError("max_actual_risk_pct 必须大于 0 或设为 None。")
    if cfg.max_chase_pct is not None and cfg.max_chase_pct <= 0:
        raise ValueError("max_chase_pct 必须大于 0 或设为 None。")


def _validate_detector_parameters(detector_name: str, cfg: StrategySuiteConfig) -> None:
    if detector_name == "trend":
        _validate_trend_parameters(cfg)
        return
    if detector_name == "range":
        _validate_range_parameters(cfg)
        return
    if detector_name == "channel":
        _validate_channel_parameters(cfg)
        return
    if detector_name == "reversal":
        _validate_reversal_parameters(cfg)
        return
    raise ValueError(f"不支持的 detector：{detector_name}")


def _validate_trend_parameters(cfg: StrategySuiteConfig) -> None:
    if cfg.trend_h2_min_pullback_legs < 1:
        raise ValueError("trend_h2_min_pullback_legs 至少需要 1。")
    if cfg.trend_pullback_lookback < 1:
        raise ValueError("trend_pullback_lookback 至少需要 1。")
    if not 0 < cfg.trend_strong_close_pos < 1:
        raise ValueError("trend_strong_close_pos 必须在 0 到 1 之间。")
    if not 0 <= cfg.trend_min_body_ratio <= 1:
        raise ValueError("trend_min_body_ratio 必须在 0 到 1 之间。")


def _validate_range_parameters(cfg: StrategySuiteConfig) -> None:
    if not 0 <= cfg.range_middle_low < cfg.range_middle_high <= 1:
        raise ValueError("range_middle_low/range_middle_high 必须在 0 到 1 之间且 low < high。")
    if cfg.range_false_break_buffer < 0:
        raise ValueError("range_false_break_buffer 不能为负数。")
    if not 0 < cfg.range_strong_close_pos < 1:
        raise ValueError("range_strong_close_pos 必须在 0 到 1 之间。")
    if cfg.range_min_score < 0:
        raise ValueError("range_min_score 不能为负数。")


def _validate_channel_parameters(cfg: StrategySuiteConfig) -> None:
    if cfg.channel_method not in {"regression", "swing"}:
        raise ValueError("channel_method 仅支持 regression 或 swing。")
    if cfg.channel_break_buffer < 0:
        raise ValueError("channel_break_buffer 不能为负数。")
    if cfg.channel_swing_left_bars < 1 or cfg.channel_swing_right_bars < 1:
        raise ValueError("channel_swing_left_bars/channel_swing_right_bars 至少需要 1。")


def _validate_reversal_parameters(cfg: StrategySuiteConfig) -> None:
    if not 0 < cfg.reversal_strong_close_pos < 1:
        raise ValueError("reversal_strong_close_pos 必须在 0 到 1 之间。")
    if not 0 <= cfg.reversal_min_body_ratio <= 1:
        raise ValueError("reversal_min_body_ratio 必须在 0 到 1 之间。")
    if cfg.reversal_old_extreme_tolerance_pct < 0:
        raise ValueError("reversal_old_extreme_tolerance_pct 不能为负数。")
