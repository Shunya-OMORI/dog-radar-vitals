"""各runの実行環境（gitコミット・主要パッケージ版・GPU情報）をrun_dirへ記録する。

査読対応で「どのコードのどのバージョンでこの結果が出たか」を追跡できるようにする。
"""
from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

_TRACKED_PACKAGES = ["torch", "numpy", "pandas", "scikit-learn", "PyYAML"]


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def capture_environment(repo_root: Path) -> dict[str, Any]:
    return {
        "git": _git_info(repo_root),
        "python_version": sys.version,
        "packages": _package_versions(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
