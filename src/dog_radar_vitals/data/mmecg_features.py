"""古典MLモデル(RR Interval予測)向けの窓単位の手作り特徴量抽出。

密な波形回帰は古典回帰に不向きなため、古典MLは「窓内の平均RR Interval[ms]」という
スカラ値予測に限定する（AGENTS.md・EXPERIMENTS.md参照）。犬用`data/features.py`と同様の
方針（分散最大のチャネルを選び時間統計量+バンドパワーに要約）だが、対象が「レンジビン」
ではなく「50点の心臓表面変位計測(RCG)チャネル」であるため、チャネル選択の基準を
心拍帯域(0.8-3Hz)のFFTパワーが最大のチャネルに変更している。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dog_radar_vitals.data.mmecg import MMECGRecording, load_trial, trial_ids_for_subjects
from dog_radar_vitals.data.rpeaks import detect_r_peaks

HR_BAND_HZ = (0.8, 3.0)  # 48-180 bpm相当

FEATURE_NAMES = [
    "signal_mean",
    "signal_std",
    "signal_min",
    "signal_max",
    "diff_std",
    "hr_band_energy",
    "hr_band_peak_hz",
]


def _select_cardiac_channel(window: np.ndarray, fs: int) -> np.ndarray:
    """window: (T, 50) -> 心拍帯域パワーが最大の1チャネル (T,)"""
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / fs)
    band_mask = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])

    best_energy, best_ch = -1.0, 0
    for ch in range(window.shape[1]):
        spectrum = np.abs(np.fft.rfft(window[:, ch] - window[:, ch].mean()))
        energy = float(spectrum[band_mask].sum()) if band_mask.any() else 0.0
        if energy > best_energy:
            best_energy, best_ch = energy, ch
    return window[:, best_ch]


def extract_window_features(window: np.ndarray, fs: int) -> np.ndarray:
    """window: (T, 50) の正規化済みRCG窓 -> (len(FEATURE_NAMES),) の特徴ベクトル。"""
    signal = _select_cardiac_channel(window, fs)
    diff = np.diff(signal, prepend=signal[0])

    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    band_mask = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    hr_energy = float(spectrum[band_mask].sum()) if band_mask.any() else 0.0
    hr_peak_hz = float(freqs[band_mask][np.argmax(spectrum[band_mask])]) if band_mask.any() else 0.0

    return np.array(
        [signal.mean(), signal.std(), signal.min(), signal.max(), diff.std(), hr_energy, hr_peak_hz],
        dtype=np.float32,
    )


def _iter_rr_windows(rec: MMECGRecording, window_sec: float, stride_sec: float):
    """(RCG窓 [T,50], 窓内平均RR Interval[ms]) を、窓内にR波が2つ以上あるものだけ返す。"""
    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    peak_indices = detect_r_peaks(ecg_filled, rec.fs)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = rec.rcg.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        peaks_in_window = peak_indices[(peak_indices >= start) & (peak_indices < end)]
        if len(peaks_in_window) < 2:
            continue
        rr_ms = float(np.diff(peaks_in_window).mean() / rec.fs * 1000.0)
        yield rec.rcg[start:end], rr_ms


def build_rr_feature_table(
    raw_root: Path, subject_ids: list[int], window_sec: float, stride_sec: float
) -> tuple[np.ndarray, np.ndarray]:
    """複数被験者の窓を特徴量化し、(X: (N, len(FEATURE_NAMES)), y: (N,)[ms]) を返す。"""
    trial_ids = trial_ids_for_subjects(raw_root, subject_ids)
    features, targets = [], []
    for trial_id in trial_ids:
        rec = load_trial(raw_root, trial_id)
        for window, rr_ms in _iter_rr_windows(rec, window_sec, stride_sec):
            features.append(extract_window_features(window, rec.fs))
            targets.append(rr_ms)
    return np.stack(features), np.array(targets, dtype=np.float32)
