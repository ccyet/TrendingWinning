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
