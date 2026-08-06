"""RCG(50ch) -> R波heatmapの窓切り出し。`rpeak_windowing.py`（Schellenberger）と対になる。"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dog_radar_vitals.data.mmecg import MMECGRecording
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, zscore_channels
from dog_radar_vitals.data.rpeaks import (
    build_peak_heatmap,
    detect_r_and_t_peaks_neurokit,
    detect_r_peaks,
    detect_r_peaks_neurokit,
)


def iter_peak_windows(
    rec: MMECGRecording,
    window_sec: float,
    stride_sec: float,
    complex_input: bool = False,
    detector: str = "legacy",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(RCG窓 [T,50](real or complex64), R波heatmap窓 [T]) を順に返す。

    detector: "legacy"(z-score+find_peaks、既定・過去との再現性のため)、
        "neurokit"(2026-08-06の恒久対応、R/T混同を回避)、
        "neurokit_rt"(2026-08-06のユーザ発案の派生案: R波だけでなくT波も教師
        heatmapに含める。モデルに「T波を無視しろ」という一貫しない教師を
        与え続けるのをやめ、R/Tどちらを検出したかの選別は後処理に任せる。
        評価時の正解はdetect_r_peaks_neurokit(R波のみ)を使うため、この
        モードの効果は学習後の後処理付き評価で確認する)。
    """
    nan_mask = None
    if np.isnan(rec.rcg).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.rcg).any(axis=1) | np.isnan(rec.ecg)

    rcg_z = zscore_channels(rec.rcg)
    rcg_input = analytic_signal_channels(rcg_z) if complex_input else rcg_z

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    if detector == "neurokit_rt":
        r_peaks, t_peaks = detect_r_and_t_peaks_neurokit(ecg_filled, rec.fs)
        peak_indices = np.sort(np.concatenate([r_peaks, t_peaks]))
    else:
        peak_fn = detect_r_peaks_neurokit if detector == "neurokit" else detect_r_peaks
        peak_indices = peak_fn(ecg_filled, rec.fs)
    heatmap = build_peak_heatmap(peak_indices, length=len(ecg_filled), fs=rec.fs)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = rcg_input.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        yield rcg_input[start:end], heatmap[start:end]
