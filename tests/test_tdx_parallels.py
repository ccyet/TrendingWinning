from __future__ import annotations

import subprocess
from pathlib import Path

from trending_winning.data.tdx_parallels import (
    ParallelsTdxConfig,
    build_parallels_tdx_command,
    mac_path_to_parallels_shared_path,
    run_parallels_tdx_command,
)


def test_mac_path_to_parallels_shared_path_maps_user_home() -> None:
    mapped = mac_path_to_parallels_shared_path(
        "/Users/a1234/Documents/TrendingWinning",
        home=Path("/Users/a1234"),
    )

    assert mapped == r"C:\Mac\Home\Documents\TrendingWinning"


def test_build_parallels_tdx_command_runs_cli_inside_windows_repo() -> None:
    command = build_parallels_tdx_command(
        config=ParallelsTdxConfig(
            vm_name="Windows 11",
            windows_python=r"C:\Users\Public\venvs\trending-winning\Scripts\python.exe",
            windows_repo=r"C:\Mac\Home\Documents\TrendingWinning",
        ),
        cli_args=[
            "tdx-doctor",
            "--runtime",
            "local",
            "--symbols",
            "000001.SZ",
            "--start",
            "2026-05-25 09:30:00",
            "--end",
            "2026-05-25 15:00:00",
            "--tdx-path",
            r"C:\new_tdx64\PYPlugins\user",
        ],
    )

    assert command[:6] == ["prlctl", "exec", "Windows 11", "cmd", "/d", "/s"]
    assert command[6] == "/c"
    assert "cd /d C:\\Mac\\Home\\Documents\\TrendingWinning" in command[7]
    assert "python.exe -m trending_winning.cli tdx-doctor --runtime local" in command[7]
    assert r"C:\new_tdx64\PYPlugins\user" in command[7]


def test_run_parallels_tdx_command_uses_prlctl_and_returns_stdout(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="status ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_parallels_tdx_command(
        config=ParallelsTdxConfig(vm_name="Windows 11", windows_python="python", windows_repo=r"C:\repo"),
        cli_args=["tdx-doctor", "--runtime", "local", "--symbols", "000001.SZ", "--start", "2026-05-25", "--end", "2026-05-25"],
    )

    assert result.returncode == 0
    assert result.stdout == "status ok\n"
    assert captured["cmd"][0:3] == ["prlctl", "exec", "Windows 11"]
    assert captured["kwargs"]["capture_output"] is True
    assert "text" not in captured["kwargs"]


def test_run_parallels_tdx_command_decodes_windows_gbk_output(monkeypatch) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 0, stdout="状态 ok\n".encode("gbk"), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_parallels_tdx_command(
        config=ParallelsTdxConfig(vm_name="Windows 11", windows_python="python", windows_repo=r"C:\repo"),
        cli_args=["tdx-doctor", "--runtime", "local", "--symbols", "000001.SZ", "--start", "2026-05-25", "--end", "2026-05-25"],
    )

    assert result.stdout == "状态 ok\n"
