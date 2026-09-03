"""ECG波形推定モデル(名前→クラス)の一元管理。

`registry.py`（HR/BR用、常にn_binsを渡す呼び出し規約）とは入出力の形が違う
（窓->スカラ ではなく 窓->波形）ため、レジストリを分けている。
"""
from __future__ import annotations

from typing import Any

from dog_radar_vitals.models.deep.ecg_cnn1d import ECGWaveformCNN1D
from dog_radar_vitals.models.deep.ecg_complex_cnn import ECGWaveformComplexCNN1D
from dog_radar_vitals.models.deep.ecg_complex_cnn_v2 import ECGWaveformComplexCNN1DV2
from dog_radar_vitals.models.deep.ecg_conformer import ECGWaveformConformer
from dog_radar_vitals.models.deep.ecg_conv_ncp import ECGConvNCP
from dog_radar_vitals.models.deep.ecg_lstm import ECGWaveformLSTM
from dog_radar_vitals.models.deep.ecg_ncp import ECGWaveformNCP
from dog_radar_vitals.models.deep.ecg_transformer import ECGWaveformTransformer
from dog_radar_vitals.models.deep.ecg_unet1d import ECGWaveformUNet1D
from dog_radar_vitals.models.deep.rpeak_cnn1d import RPeakCNN1D
from dog_radar_vitals.models.deep.rpeak_complex_cnn import RPeakComplexCNN1D
from dog_radar_vitals.models.deep.rpeak_complex_cnn_v2 import RPeakComplexCNN1DV2
from dog_radar_vitals.models.deep.rpeak_conformer import RPeakConformer
from dog_radar_vitals.models.deep.rpeak_conv_ncp import RPeakConvNCP
from dog_radar_vitals.models.deep.rpeak_lstm import RPeakLSTM
from dog_radar_vitals.models.deep.rpeak_attention_unet1d import RPeakAttentionUNet1D
from dog_radar_vitals.models.deep.rpeak_ncp import RPeakNCP
from dog_radar_vitals.models.deep.rpeak_transformer import RPeakTransformer
from dog_radar_vitals.models.deep.rpeak_unet1d import RPeakUNet1D

ECG_MODEL_REGISTRY: dict[str, type] = {
    "ecg_cnn1d": ECGWaveformCNN1D,
    "rpeak_cnn1d": RPeakCNN1D,
    "ecg_transformer": ECGWaveformTransformer,
    "rpeak_transformer": RPeakTransformer,
    "ecg_lstm": ECGWaveformLSTM,
    "rpeak_lstm": RPeakLSTM,
    "ecg_ncp": ECGWaveformNCP,
    "rpeak_ncp": RPeakNCP,
    "ecg_complex_cnn": ECGWaveformComplexCNN1D,
    "rpeak_complex_cnn": RPeakComplexCNN1D,
    "ecg_complex_cnn_v2": ECGWaveformComplexCNN1DV2,
    "rpeak_complex_cnn_v2": RPeakComplexCNN1DV2,
    "ecg_conformer": ECGWaveformConformer,
    "rpeak_conformer": RPeakConformer,
    "ecg_unet1d": ECGWaveformUNet1D,
    "rpeak_unet1d": RPeakUNet1D,
    "rpeak_attention_unet1d": RPeakAttentionUNet1D,
    "ecg_conv_ncp": ECGConvNCP,
    "rpeak_conv_ncp": RPeakConvNCP,
}


def build_ecg_model(name: str, **kwargs: Any):
    if name not in ECG_MODEL_REGISTRY:
        raise ValueError(f"unknown ECG model '{name}'. known models: {sorted(ECG_MODEL_REGISTRY)}")
    return ECG_MODEL_REGISTRY[name](**kwargs)
