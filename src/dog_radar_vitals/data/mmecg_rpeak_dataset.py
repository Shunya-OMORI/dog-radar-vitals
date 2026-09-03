"""RCG窓とR波heatmap窓を束ねるPyTorch Dataset。`rpeak_dataset.py`（Schellenberger）と対になる。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.mmecg import load_trial
from dog_radar_vitals.data.mmecg_rpeak_windowing import iter_peak_windows


class MMECGRPeakWindowDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        trial_ids: list[int],
        window_sec: float,
        stride_sec: float,
        complex_input: bool = False,
        rpeak_detector: str = "legacy",
        heatmap_sigma_ms: float = 10.0,
        normalize: str = "zscore",
        target_mode: str = "all_peaks",
        target_shape: str = "gaussian",
        box_half_width_ms: float = 150.0,
    ) -> None:
        # 2026-08-28: 元々このクラスはiter_peak_windowsの既定値(detector="legacy",
        # sigma_ms=10.0)を暗黙に使っており、config284(neurokit, sigma=15ms)とは
        # 教師の作り方が食い違っていた。312/313等の「config284と単一変数だけ違う」
        # という比較の前提が崩れていたため、明示的に渡せるようにした。
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        for trial_id in trial_ids:
            rec = load_trial(raw_root, trial_id)
            self._samples.extend(iter_peak_windows(
                rec, window_sec, stride_sec, complex_input=complex_input,
                detector=rpeak_detector, heatmap_sigma_ms=heatmap_sigma_ms, normalize=normalize,
                target_mode=target_mode, target_shape=target_shape, box_half_width_ms=box_half_width_ms,
            ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        x_tensor = torch.from_numpy(x)
        x_tensor = x_tensor.to(torch.complex64) if np.iscomplexobj(x) else x_tensor.float()
        return x_tensor, torch.from_numpy(y).float()
