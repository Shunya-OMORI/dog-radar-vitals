"""YAML設定の読み込み。`extends: <相対パス>` で親設定を1階層だけ継承できる。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    if "extends" in config:
        parent_path = (path.parent / config["extends"]).resolve()
        parent = load_config(parent_path)
        config = _deep_merge(parent, config)

    return config
