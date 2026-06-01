from __future__ import annotations

from pathlib import Path


def test_source_code_has_no_akshare_channel() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_paths = list((root / "trending_winning").rglob("*.py"))
    checked_paths.extend([root / "requirements.txt", root / "pyproject.toml"])

    offenders = [
        str(path.relative_to(root))
        for path in checked_paths
        if path.exists() and "akshare" in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []


def test_core_runtime_does_not_use_dataframe_iterrows() -> None:
    root = Path(__file__).resolve().parents[1]
    checked_paths = list((root / "trending_winning").rglob("*.py"))

    offenders = [
        str(path.relative_to(root))
        for path in checked_paths
        if path.exists() and ".iterrows(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_000852_strategy_guides_are_actionable_html() -> None:
    root = Path(__file__).resolve().parents[1]
    guide_paths = [
        root / "docs" / "trend_strategy_000852_guide.html",
        root / "docs" / "channel_strategy_000852_guide.html",
    ]

    for path in guide_paths:
        assert path.exists(), f"缺少策略讲解 HTML：{path.name}"
        html = path.read_text(encoding="utf-8")
        for keyword in (
            "000852.SH",
            "默认参数",
            "触发条件拆解",
            "背景条件",
            "信号K条件",
            "订单条件",
            "风控条件",
            "信号分类",
            "或然情况",
            "或然情况分层",
            "信号生命周期",
            "观察信号",
            "触发后拒单",
            "触发矩阵",
            "有效但未触发",
            "过滤拒单",
            "撮合拒单",
            "策略空间",
            "策略空间不是无限可做",
            "可交易空间",
            "参数空间",
            "订单空间",
            "风险空间",
            "持仓空间",
            "过滤空间",
            "统计空间",
            "失效空间",
            "执行边界",
            "背景不满足",
            "信号成立未触发",
            "触发后风险不合格",
            "早期顺势",
            "中段回撤",
            "末端衰竭",
            "信号K",
            "挂单",
            "止损",
            "退出",
            "开仓量化规则",
            "因子计算方式",
            "可能性展示",
            "虚线挂单",
            "虚线止损",
            "虚线目标",
            "合并卡片",
            "entry_rule_card",
            "factor_calc_grid",
            "factor-kline",
            "merged-rule-grid",
            "merged-rule-card",
            "背景因子合并",
            "信号K因子合并",
            "执行风控合并",
            "scenario-strip",
            "开仓决策流水线",
            "因子核对表",
            "计算公式",
            "默认判定",
            "盘面读法",
            "失败分支",
            "可能性矩阵",
            "正常开仓",
            "有效未触发",
            "风险拒单",
            "过滤观察",
            "decision-pipeline",
            "factor-audit-grid",
            "scenario-matrix",
            "<svg",
        ):
            assert keyword in html, f"{path.name} 缺少 {keyword}"
        assert html.count("factor-kline") >= 6, f"{path.name} 因子 K 线示意不足"
        assert html.count("factor-audit-card") >= 6, f"{path.name} 因子核对卡不足"
        assert html.count("scenario-matrix-card") >= 4, f"{path.name} 可能性 K 线场景不足"

    trend_html = guide_paths[0].read_text(encoding="utf-8")
    for keyword in (
        "TrendScore = slope_z + structure_score + ma_alignment + close_strength + follow_through",
        "close_pos = (close - low) / max(high - low, eps)",
        "body_ratio = abs(close - open) / max(high - low, eps)",
        "entry_long = signal_high + tick",
        "stop_long = min(low[-pullback_lookback:]) - tick",
        "target_long = entry_long + risk_reward * (entry_long - stop_long)",
        "趋势背景：斜率、结构、均线",
        "信号质量：收盘位置、实体、H/L 腿",
        "执行边界：挂单、止损、目标",
        "H2/L2 二次顺势入场",
        "早期顺势 / 中段回撤 / 末端衰竭",
    ):
        assert keyword in trend_html, f"{guide_paths[0].name} 缺少 {keyword}"

    channel_html = guide_paths[1].read_text(encoding="utf-8")
    for keyword in (
        "channel_mid = rolling_regression(log(close), lookback=40)",
        "channel_upper = channel_mid + sigma_multiple * residual_std",
        "channel_pos = (close - channel_mid) / max(channel_upper - channel_mid, eps)",
        "break_up = close > prior_channel_upper + channel_break_buffer",
        "entry_long = signal_high + tick",
        "stop_long = prior_channel_lower - tick",
        "通道背景：中线、宽度、R²",
        "突破质量：通道位置、突破距离、收盘确认",
        "执行边界：挂单、对侧止损、2R",
        "通道内 / 上轨突破 / 末端假突破",
    ):
        assert keyword in channel_html, f"{guide_paths[1].name} 缺少 {keyword}"
