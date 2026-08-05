"""Schellenberger et al. (2020)の臨床ECG(ヒト30名、レーダ非依存)から単一拍を切り出すDataset。

ユーザ提案の「ECGの基盤モデル/事前学習モデル」路線の実装。ミリ波レーダとは無関係な、
クリーンな臨床ECGだけを使って「PQRST波形の形」を教師なしで学習させる
（`models/deep/ecg_beat_autoencoder.py`のオートエンコーダの事前学習に使う）。
レーダを一切使わないため、MMECGのtrain/val/test分割とは独立しており、リークの懸念は無い
（無関係な公開データセットで事前学習し、対象タスクにfine-tuningするという一般的な転移学習の
構図。詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from dog_radar_vitals.data.rpeaks import detect_r_peaks
from dog_radar_vitals.data.schellenberger import HumanRecording, list_subjects, load_recording

T_FIXED_ECG = 200  # SCEGの単一拍出力長と合わせる


def _extract_beats(rec: HumanRecording) -> list[np.ndarray]:
    ecg = rec.ecg.astype(np.float32)
    ecg = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
    peaks = detect_r_peaks(ecg, rec.fs)
    if len(peaks) < 2:
        return []

    boundaries = [0] + [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)] + [len(ecg)]
    beats = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 10 or np.isnan(ecg[start:end]).any():
            continue
        beats.append(resample(ecg[start:end], T_FIXED_ECG).astype(np.float32))
    return beats


class ECGBeatDataset(Dataset):
    def __init__(self, raw_root: Path, subject_ids: list[str] | None = None, scenario: str = "Resting") -> None:
        subject_ids = subject_ids or list_subjects(raw_root)
        self._beats: list[np.ndarray] = []
        for subject_id in subject_ids:
            rec = load_recording(raw_root, subject_id, scenario)
            self._beats.extend(_extract_beats(rec))

    def __len__(self) -> int:
        return len(self._beats)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self._beats[idx])
