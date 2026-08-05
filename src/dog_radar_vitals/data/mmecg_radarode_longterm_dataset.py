"""radarODEの長期再構成ネットワーク学習用: RCG窓+SCEG由来の"morphological reference"+ECG窓。

学習済み(凍結)の`RadarODESCEG`を使い、`mmecg_ppi.py`のPPI推定で区切った各心拍について
単一拍ECGを生成し、元の時間軸に貼り戻して録音全体の長さの連続信号
（morphological reference、Fig.2(c)）を作る。これを4秒窓に切り出し、同じ窓のRCG・
真のECGと組にして`RadarODELongTerm`（`models/deep/radarode_longterm.py`）に渡す。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset

from dog_radar_vitals.data.mmecg import MMECGRecording, load_trial
from dog_radar_vitals.data.mmecg_singlecycle_dataset import iter_single_cycles_with_bounds
from dog_radar_vitals.data.mmecg_windowing import zscore_channels
from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap


@torch.no_grad()
def build_morphological_reference(
    raw_root: Path, trial_id: int, sceg_model: torch.nn.Module, device: torch.device
) -> tuple[MMECGRecording, np.ndarray]:
    """録音全体の長さに揃えたmorphological reference(np.ndarray, z-score空間)を返す。"""
    rec = load_trial(raw_root, trial_id)
    reference = np.zeros_like(rec.ecg, dtype=np.float32)

    sceg_model.eval()
    for sst_seg, _, ecg_start, ecg_end in iter_single_cycles_with_bounds(raw_root, trial_id):
        x = torch.from_numpy(sst_seg.astype(np.float32)).unsqueeze(0).to(device)
        pred_cycle = sceg_model(x).squeeze(0).cpu().numpy()  # (T_FIXED_ECG,)
        resampled = resample(pred_cycle, ecg_end - ecg_start)
        reference[ecg_start:ecg_end] = resampled

    return rec, reference


class MMECGRadarODELongtermDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        trial_ids: list[int],
        window_sec: float,
        stride_sec: float,
        sceg_model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for trial_id in trial_ids:
            rec, reference = build_morphological_reference(raw_root, trial_id, sceg_model, device)

            nan_mask = None
            if np.isnan(rec.rcg).any() or np.isnan(rec.ecg).any():
                nan_mask = np.isnan(rec.rcg).any(axis=1) | np.isnan(rec.ecg)

            rcg_z = zscore_channels(rec.rcg)  # (T, 50)
            ecg_z = zscore_with_nan_gap(rec.ecg)  # (T,)

            window_len = int(window_sec * rec.fs)
            stride = int(stride_sec * rec.fs)
            n_steps = rcg_z.shape[0]
            for start in range(0, n_steps - window_len + 1, stride):
                end = start + window_len
                if nan_mask is not None and nan_mask[start:end].any():
                    continue
                self._samples.append(
                    (rcg_z[start:end].T.astype(np.float32), reference[start:end], ecg_z[start:end].astype(np.float32))
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rcg, ref, ecg = self._samples[idx]
        return torch.from_numpy(rcg), torch.from_numpy(ref), torch.from_numpy(ecg)
