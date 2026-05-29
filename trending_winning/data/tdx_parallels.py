from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess

PARALLELS_TDX_VM_ENV_VAR = "TDX_PARALLELS_VM"
PARALLELS_TDX_PYTHON_ENV_VAR = "TDX_PARALLELS_PYTHON"
PARALLELS_TDX_REPO_ENV_VAR = "TDX_PARALLELS_REPO"
DEFAULT_PARALLELS_VM = "Windows 11"
DEFAULT_WINDOWS_PYTHON = r"C:\Users\Public\venvs\trending-winning\Scripts\python.exe"


@dataclass(frozen=True)
class ParallelsTdxConfig:
    """Parallels/Windows TDX 执行环境；Mac 只负责调度，真实 tqcenter 在 Windows 内运行。"""

    vm_name: str
    windows_python: str
    windows_repo: str


def default_parallels_tdx_config(*, cwd: Path | None = None) -> ParallelsTdxConfig:
    """从环境变量和当前仓库路径推导默认 Parallels TDX 运行配置。"""
    repo_path = os.getenv(PARALLELS_TDX_REPO_ENV_VAR, "").strip()
    if not repo_path:
        repo_path = mac_path_to_parallels_shared_path(str(cwd or Path.cwd()))
    return ParallelsTdxConfig(
        vm_name=os.getenv(PARALLELS_TDX_VM_ENV_VAR, DEFAULT_PARALLELS_VM).strip() or DEFAULT_PARALLELS_VM,
        windows_python=os.getenv(PARALLELS_TDX_PYTHON_ENV_VAR, DEFAULT_WINDOWS_PYTHON).strip()
        or DEFAULT_WINDOWS_PYTHON,
        windows_repo=repo_path,
    )


def mac_path_to_parallels_shared_path(value: str, *, home: Path | None = None) -> str:
    """把 macOS 用户目录映射到 Parallels 默认共享盘路径；Windows 路径和相对路径原样返回。"""
    text = str(value).strip().strip('"')
    if not text or _looks_like_windows_path(text) or not text.startswith("/"):
        return text
    home_path = home or Path.home()
    path = Path(text).expanduser()
    try:
        relative = path.relative_to(home_path)
    except ValueError:
        return text
    parts = ["C:", "Mac", "Home", *relative.parts]
    return "\\".join(parts)


def build_parallels_tdx_command(*, config: ParallelsTdxConfig, cli_args: list[str]) -> list[str]:
    """构造 prlctl 命令；内部固定切回 local runtime，避免递归再次调度 Parallels。"""
    inner_args = [config.windows_python, "-m", "trending_winning.cli", *cli_args]
    inner_command = f"cd /d {_quote_windows_arg(config.windows_repo)} && {subprocess.list2cmdline(inner_args)}"
    return ["prlctl", "exec", config.vm_name, "cmd", "/d", "/s", "/c", inner_command]


def run_parallels_tdx_command(*, config: ParallelsTdxConfig, cli_args: list[str]) -> subprocess.CompletedProcess[str]:
    """通过 prlctl 在 Windows VM 内执行 TDX CLI 命令，并把 stdout/stderr 返回给调用方。"""
    command = build_parallels_tdx_command(config=config, cli_args=cli_args)
    result = subprocess.run(command, capture_output=True, check=False)
    return subprocess.CompletedProcess(
        args=result.args,
        returncode=result.returncode,
        stdout=_decode_windows_output(result.stdout),
        stderr=_decode_windows_output(result.stderr),
    )


def _looks_like_windows_path(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith("\\\\")


def _quote_windows_arg(value: str) -> str:
    return subprocess.list2cmdline([value])


def _decode_windows_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    for encoding in ("gbk", "cp936", "utf-8"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            continue
    return value.decode("utf-8", errors="replace")
