"""深層モデル(nn.Module)の名前→クラスの一元管理。新モデル追加時はここに1行足すだけでよい。"""
from __future__ import annotations

from typing import Any

from dog_radar_vitals.models.deep.cnn1d import VitalsCNN1D
from dog_radar_vitals.models.deep.lstm import VitalsLSTM
from dog_radar_vitals.models.deep.transformer import VitalsTransformer

DEEP_MODEL_REGISTRY: dict[str, type] = {
    "transformer": VitalsTransformer,
    "cnn1d": VitalsCNN1D,
    "lstm": VitalsLSTM,
}


def build_deep_model(name: str, **kwargs: Any):
    if name not in DEEP_MODEL_REGISTRY:
        raise ValueError(f"unknown deep model '{name}'. known models: {sorted(DEEP_MODEL_REGISTRY)}")
    return DEEP_MODEL_REGISTRY[name](**kwargs)
