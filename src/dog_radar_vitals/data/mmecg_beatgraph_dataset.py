"""拍単位のRCG segment -> PQRST 5点グラフ を束ねるPyTorch Dataset（GNN/GANモデル用）。

正解のP/Q/S/T位置は`neurokit2.ecg_delineate`（method='peak'）による自動検出であり、
臨床アノテーションではない疑似正解である点に注意（ユーザ合意事項、AGENTS.md参照）。
R波は`data/rpeaks.py`ではなくneurokit2の`ecg_peaks`を使う（delineateがrpeaks_infoの
形式を要求するため、同じneurokit2の検出器で揃える）。

各拍について、R波を中心に`segment_sec`幅のRCG(50ch)を切り出し入力とする。ターゲットは
5ノード(P,Q,R,S,T)×2特徴（Rからの相対時刻[正規化済み]、z-score済みECG振幅）。
ノード順序・エッジ(P-Q-R-S-T鎖状、無向)は`models/deep/beatgraph_gnn.py`と共有する定数として
`BEAT_GRAPH_NODE_ORDER`をこのモジュールに置く。
"""
from __future__ import annotations

from pathlib import Path

import neurokit2 as nk
import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.mmecg import MMECGRecording, load_trial
from dog_radar_vitals.data.mmecg_windowing import zscore_channels

BEAT_GRAPH_NODE_ORDER = ["P", "Q", "R", "S", "T"]
TIME_NORM_MS = 200.0  # 時刻オフセットの正規化スケール(ms)。1拍の半周期程度を目安に選んだ。


def _delineate_beats(rec: MMECGRecording) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """neurokit2でR波検出+PQRST delineationを行い、(rpeaks, waves)を返す。"""
    ecg = rec.ecg.astype(np.float64)
    cleaned = nk.ecg_clean(ecg, sampling_rate=rec.fs)
    _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=rec.fs)
    rpeaks = rpeaks_info["ECG_R_Peaks"]
    _, waves = nk.ecg_delineate(cleaned, rpeaks_info, sampling_rate=rec.fs, method="peak")
    return rpeaks, waves


def iter_beat_graphs(rec: MMECGRecording, segment_sec: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """(RCG segment [T,50], target graph [5,2]) のリストを返す。

    P/Q/S/Tのいずれかが未検出(NaN)、またはsegmentが録音境界からはみ出す拍はスキップする。
    """
    rpeaks, waves = _delineate_beats(rec)
    ecg_z = (rec.ecg - np.nanmean(rec.ecg)) / (np.nanstd(rec.ecg) + 1e-8)
    rcg_z = zscore_channels(rec.rcg)

    segment_len = int(segment_sec * rec.fs)
    half = segment_len // 2

    p_peaks = np.asarray(waves["ECG_P_Peaks"], dtype=float)
    q_peaks = np.asarray(waves["ECG_Q_Peaks"], dtype=float)
    s_peaks = np.asarray(waves["ECG_S_Peaks"], dtype=float)
    t_peaks = np.asarray(waves["ECG_T_Peaks"], dtype=float)

    samples = []
    for i, r_idx in enumerate(rpeaks):
        landmark_idx = {"P": p_peaks[i], "Q": q_peaks[i], "R": float(r_idx), "S": s_peaks[i], "T": t_peaks[i]}
        if any(np.isnan(v) for v in landmark_idx.values()):
            continue

        start, end = int(r_idx) - half, int(r_idx) - half + segment_len
        if start < 0 or end > rcg_z.shape[0]:
            continue

        node_features = []
        for name in BEAT_GRAPH_NODE_ORDER:
            idx = int(round(landmark_idx[name]))
            time_offset_ms = (landmark_idx[name] - r_idx) / rec.fs * 1000.0
            time_offset_norm = time_offset_ms / TIME_NORM_MS
            amplitude = float(ecg_z[idx])
            node_features.append([time_offset_norm, amplitude])

        samples.append((rcg_z[start:end], np.array(node_features, dtype=np.float32)))
    return samples


class MMECGBeatGraphDataset(Dataset):
    def __init__(self, raw_root: Path, trial_ids: list[int], segment_sec: float = 0.8) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        for trial_id in trial_ids:
            rec = load_trial(raw_root, trial_id)
            self._samples.extend(iter_beat_graphs(rec, segment_sec))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
