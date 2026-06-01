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
            "过滤空间",
            "统计空间",
            "失效空间",
            "执行边界",
            "信号K",
            "挂单",
            "止损",
            "退出",
            "<svg",
        ):
            assert keyword in html, f"{path.name} 缺少 {keyword}"
