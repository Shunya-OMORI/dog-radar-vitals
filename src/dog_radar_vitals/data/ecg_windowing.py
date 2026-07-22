"""レーダI/Q -> ECG波形の窓切り出し（同一サンプリングレート・同一長のsequence-to-sequence）。

犬データセットの `windowing.py` は「窓 -> 1Hzのスカラ値」（多対一）だったのに対し、
こちらは「窓 -> 同じ長さの波形」（多対多）である点が本質的に異なるため別ファイルに分離する。
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dog_radar_vitals.data.schellenberger import HumanRecording


def zscore_with_nan_gap(x: np.ndarray) -> np.ndarray:
    # nanmean/nanstdを使うことで、録音中の一部にNaN欠損があっても正規化統計量そのものが
    # NaN化するのを避ける（該当区間を含む窓自体は呼び出し元(iter_windows)でスキップする）。
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def iter_windows(rec: HumanRecording, window_sec: float, stride_sec: float) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(radar_iq窓 [T,2], ecg窓 [T]) を順に返す。両方とも録音単位でz-score正規化する。

    センサの瞬断による欠損(NaN)が短い区間に発生することがある（実例: GDN0003のECGに
    41サンプル=20msのNaN区間）。z-score正規化はNaNが1つでもあると平均・分散ごとNaN化し
    録音全体を汚染するため、正規化前の生値でNaNを検出し、該当窓は丸ごとスキップする。
    """
    if np.isnan(rec.radar_i).any() or np.isnan(rec.radar_q).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.radar_i) | np.isnan(rec.radar_q) | np.isnan(rec.ecg)
    else:
        nan_mask = None

    radar_iq = np.stack([zscore_with_nan_gap(rec.radar_i), zscore_with_nan_gap(rec.radar_q)], axis=-1)
    ecg = zscore_with_nan_gap(rec.ecg)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = radar_iq.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        yield radar_iq[start:end], ecg[start:end]
