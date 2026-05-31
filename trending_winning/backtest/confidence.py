from __future__ import annotations

import math

import pandas as pd


CONFIDENCE_Z_SCORE = 1.96

SAMPLE_CONFIDENCE_STAT_KEYS = (
    "win_rate_ci_lower",
    "win_rate_ci_upper",
    "avg_return_standard_error",
    "avg_return_ci_lower",
    "avg_return_ci_upper",
    "positive_expectancy_probability",
)


def sample_confidence_statistics(returns: pd.Series) -> dict[str, float]:
    """计算逐笔收益的小样本稳定性指标，避免只用点估计判断策略质量。"""
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return {key: 0.0 for key in SAMPLE_CONFIDENCE_STAT_KEYS}

    avg_return = float(clean.mean())
    return_standard_error = standard_error(clean)
    win_rate = float((clean > 0).mean())
    win_rate_ci_lower, win_rate_ci_upper = wilson_score_interval(win_rate, len(clean))
    if return_standard_error > 0:
        avg_return_ci_lower = avg_return - CONFIDENCE_Z_SCORE * return_standard_error
        avg_return_ci_upper = avg_return + CONFIDENCE_Z_SCORE * return_standard_error
    else:
        avg_return_ci_lower = avg_return
        avg_return_ci_upper = avg_return

    return {
        "win_rate_ci_lower": _round_float(win_rate_ci_lower),
        "win_rate_ci_upper": _round_float(win_rate_ci_upper),
        "avg_return_standard_error": _round_float(return_standard_error),
        "avg_return_ci_lower": _round_float(avg_return_ci_lower),
        "avg_return_ci_upper": _round_float(avg_return_ci_upper),
        "positive_expectancy_probability": _round_float(
            positive_expectancy_probability(avg_return, return_standard_error)
        ),
    }


def standard_error(values: pd.Series) -> float:
    """计算样本均值标准误；样本不足时返回 0，避免制造虚假精度。"""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1)) / math.sqrt(len(clean))


def wilson_score_interval(rate: float, sample_count: int) -> tuple[float, float]:
    """用 Wilson 区间估计胜率范围，比小样本正态近似更稳。"""
    if sample_count <= 0:
        return 0.0, 0.0
    z2 = CONFIDENCE_Z_SCORE**2
    denominator = 1.0 + z2 / sample_count
    center = rate + z2 / (2.0 * sample_count)
    margin = CONFIDENCE_Z_SCORE * math.sqrt(rate * (1.0 - rate) / sample_count + z2 / (4.0 * sample_count**2))
    return max(0.0, (center - margin) / denominator), min(1.0, (center + margin) / denominator)


def positive_expectancy_probability(avg_return: float, return_standard_error: float) -> float:
    """估计真实均值大于 0 的概率；标准误为 0 时按确定性收益处理。"""
    if return_standard_error <= 0:
        if avg_return > 0:
            return 1.0
        if avg_return < 0:
            return 0.0
        return 0.5
    z_value = avg_return / return_standard_error
    return 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))


def _round_float(value: float) -> float:
    return float(round(float(value), 12))
