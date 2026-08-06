"""RCG窓・ECG波形窓に加え、各点の3D座標(posXYZ)も返すDataset（空間融合モデル用）。

Chen et al. (2022)のアーキテクチャは、50点の心臓動き計測を「空間分布を持つ点群」として扱い、
各点の3D位置埋め込みをTransformerで時間特徴と融合している。既存の`MMECGWindowDataset`は
posXYZを返さない（`ecg_cnn1d`等の既存モデルは50chを通常の畳み込みチャネルとしてしか
扱わないため）。空間融合モデル(`models/deep/ecg_spatial_fusion.py`)専用に、posXYZ
（トライアル内で一定、トライアルごとに異なる）を各窓に付与して返す。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.mmecg import load_trial
from dog_radar_vitals.data.mmecg_rpeak_windowing import iter_peak_windows
from dog_radar_vitals.data.mmecg_windowing import iter_windows


class MMECGSpatialWindowDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        trial_ids: list[int],
        window_sec: float,
        stride_sec: float,
        heatmap: bool = False,
        normalization: str = "zscore",
        rpeak_detector: str = "legacy",
        apply_bandpass: bool = False,
    ) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        window_fn = iter_peak_windows if heatmap else iter_windows
        window_kwargs = (
            {"detector": rpeak_detector, "apply_bandpass": apply_bandpass} if heatmap
            else {"normalization": normalization}
        )
        for trial_id in trial_ids:
            rec = load_trial(raw_root, trial_id)
            for rcg_win, target_win in window_fn(rec, window_sec, stride_sec, complex_input=False, **window_kwargs):
                self._samples.append((rcg_win, rec.posxyz, target_win))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rcg_win, posxyz, ecg_win = self._samples[idx]
        return torch.from_numpy(rcg_win).float(), torch.from_numpy(posxyz).float(), torch.from_numpy(ecg_win).float()
