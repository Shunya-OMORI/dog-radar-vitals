"""レーダI/Q窓とR波heatmap窓を束ねるPyTorch Dataset。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.rpeak_windowing import iter_peak_windows
from dog_radar_vitals.data.schellenberger import load_recording


class RPeakWindowDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        subject_ids: list[str],
        scenario: str,
        window_sec: float,
        stride_sec: float,
    ) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        for subject_id in subject_ids:
            rec = load_recording(raw_root, subject_id, scenario)
            self._samples.extend(iter_peak_windows(rec, window_sec, stride_sec))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
