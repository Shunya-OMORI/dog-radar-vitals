"""レーダI/Q -> R波heatmapの窓切り出し。

`ecg_windowing.py`（窓 -> 密なECG波形）と入出力の形は同じ(窓->同じ長さの系列)だが、
ターゲットが「R波位置に立てたガウシアンheatmap」という疎なイベント表現である点が異なる。
録音全体のheatmapを一度だけ構築してから窓切り出しする（窓ごとにR波検出をやり直すと
窓境界でR波を取りこぼすため）。
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap
from dog_radar_vitals.data.rpeaks import build_peak_heatmap, detect_r_peaks
from dog_radar_vitals.data.schellenberger import HumanRecording


def iter_peak_windows(rec: HumanRecording, window_sec: float, stride_sec: float) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(radar_iq窓 [T,2], R波heatmap窓 [T]) を順に返す。"""
    if np.isnan(rec.radar_i).any() or np.isnan(rec.radar_q).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.radar_i) | np.isnan(rec.radar_q) | np.isnan(rec.ecg)
    else:
        nan_mask = None

    radar_iq = np.stack([zscore_with_nan_gap(rec.radar_i), zscore_with_nan_gap(rec.radar_q)], axis=-1)

    # R波検出はNaN混じりだと誤検出するため、NaN区間を平均値で一時穴埋めしてから検出する
    # （検出後、該当区間を含む窓自体はNaN mask依存で除外するので実害は無い）。
    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    peak_indices = detect_r_peaks(ecg_filled, rec.fs)
    heatmap = build_peak_heatmap(peak_indices, length=len(ecg_filled), fs=rec.fs)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = radar_iq.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        yield radar_iq[start:end], heatmap[start:end]
