"""ECG波形推定モデル(名前→クラス)の一元管理。

`registry.py`（HR/BR用、常にn_binsを渡す呼び出し規約）とは入出力の形が違う
（窓->スカラ ではなく 窓->波形）ため、レジストリを分けている。
"""
from __future__ import annotations

from typing import Any

from dog_radar_vitals.models.deep.ecg_cnn1d import ECGWaveformCNN1D
from dog_radar_vitals.models.deep.rpeak_cnn1d import RPeakCNN1D

ECG_MODEL_REGISTRY: dict[str, type] = {
    "ecg_cnn1d": ECGWaveformCNN1D,
    "rpeak_cnn1d": RPeakCNN1D,
}


def build_ecg_model(name: str, **kwargs: Any):
    if name not in ECG_MODEL_REGISTRY:
        raise ValueError(f"unknown ECG model '{name}'. known models: {sorted(ECG_MODEL_REGISTRY)}")
    return ECG_MODEL_REGISTRY[name](**kwargs)
