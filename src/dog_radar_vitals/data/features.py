"""古典MLモデル向けの窓単位の手作り特徴量抽出。

深層モデルは生の窓 (window_sec*50, 467) をそのまま入力できるが、古典MLモデルに
そのままの次元を渡すとサンプル数に対して次元が大きすぎ、比較として不公平かつ
過学習しやすい。そこでAhmed et al. (2024) 自身の解析方針（分散でイヌの位置を
決め、呼吸帯・心拍帯のバンドパスで周波数を定量化する）を踏襲した特徴量に落とす。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dog_radar_vitals.data.scenario1 import RADAR_FS_HZ, load_recording
from dog_radar_vitals.data.windowing import Task, iter_windows

N_TARGET_BINS = 5
RESP_BAND_BPM = (10.0, 40.0)
HR_BAND_BPM = (60.0, 160.0)

FEATURE_NAMES = [
    "signal_mean",
    "signal_std",
    "signal_min",
    "signal_max",
    "diff_std",
    "resp_band_energy",
    "resp_band_peak_bpm",
    "hr_band_energy",
    "hr_band_peak_bpm",
]


def _band_stats(freqs: np.ndarray, spectrum: np.ndarray, band_bpm: tuple[float, float]) -> tuple[float, float]:
    mask = (freqs >= band_bpm[0] / 60.0) & (freqs <= band_bpm[1] / 60.0)
    if not mask.any():
        return 0.0, 0.0
    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]
    energy = float(band_spectrum.sum())
    peak_bpm = float(band_freqs[np.argmax(band_spectrum)] * 60.0)
    return energy, peak_bpm


def extract_window_features(window: np.ndarray, fs: int = RADAR_FS_HZ) -> np.ndarray:
    """window: (T, n_bins) の正規化済みレーダ窓 -> (len(FEATURE_NAMES),) の特徴ベクトル。"""
    variances = window.var(axis=0)
    top_bins = np.argsort(variances)[-N_TARGET_BINS:]
    signal = window[:, top_bins].mean(axis=1)
    diff = np.diff(signal, prepend=signal[0])

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))

    resp_energy, resp_peak_bpm = _band_stats(freqs, spectrum, RESP_BAND_BPM)
    hr_energy, hr_peak_bpm = _band_stats(freqs, spectrum, HR_BAND_BPM)

    return np.array(
        [
            signal.mean(),
            signal.std(),
            signal.min(),
            signal.max(),
            diff.std(),
            resp_energy,
            resp_peak_bpm,
            hr_energy,
            hr_peak_bpm,
        ],
        dtype=np.float32,
    )


def build_feature_table(
    raw_root: Path, dog_ids: list[str], task: Task, window_sec: int, stride_sec: int
) -> tuple[np.ndarray, np.ndarray]:
    """複数犬の窓を特徴量化し、(X: (N, len(FEATURE_NAMES)), y: (N,)) を返す。"""
    features, targets = [], []
    for dog_id in dog_ids:
        rec = load_recording(raw_root, dog_id)
        for window, target in iter_windows(rec, task, window_sec, stride_sec):
            features.append(extract_window_features(window))
            targets.append(target)
    return np.stack(features), np.array(targets, dtype=np.float32)
