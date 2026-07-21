"""モデル名からクラスを引くだけの一元管理。新モデル追加時はここに1行足すだけでよい。"""
from __future__ import annotations

from typing import Any

from dog_radar_vitals.models.transformer import VitalsTransformer

MODEL_REGISTRY: dict[str, type] = {
    "transformer": VitalsTransformer,
}


def build_model(name: str, **kwargs: Any):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'. known models: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
