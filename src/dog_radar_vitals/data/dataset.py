"""生のレーダ窓をそのまま入力とする深層モデル向けDataset。

犬ごとに正規化・窓切り出しを行うのは `windowing.py` の責務。ここは複数犬の窓を
束ねてPyTorchの `Dataset` インタフェースに載せるだけに留める。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.scenario1 import load_recording
from dog_radar_vitals.data.windowing import Task, iter_windows


class WindowedVitalsDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dog_ids: list[str],
        task: Task,
        window_sec: int = 10,
        stride_sec: int = 1,
    ) -> None:
        self.task = task
        self.window_sec = window_sec
        self.stride_sec = stride_sec

        self._samples: list[tuple[np.ndarray, float]] = []
        for dog_id in dog_ids:
            rec = load_recording(raw_root, dog_id)
            self._samples.extend(iter_windows(rec, task, window_sec, stride_sec))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)
