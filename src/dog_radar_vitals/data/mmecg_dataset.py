"""RCG窓とECG波形窓を束ねるPyTorch Dataset。`ecg_dataset.py`（Schellenberger）と対になる。

トライアルIDのリストを受け取る（被験者IDではない）。被験者単位でのsplit構築は
`data/mmecg.trial_ids_for_subjects`で呼び出し側が行う（同一被験者の複数トライアルが
splitを跨がないようにするのは呼び出し側の責務）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.mmecg import load_trial
from dog_radar_vitals.data.mmecg_windowing import iter_windows


class MMECGWindowDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        trial_ids: list[int],
        window_sec: float,
        stride_sec: float,
        complex_input: bool = False,
        normalization: str = "zscore",
    ) -> None:
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        for trial_id in trial_ids:
            rec = load_trial(raw_root, trial_id)
            self._samples.extend(
                iter_windows(rec, window_sec, stride_sec, complex_input=complex_input, normalization=normalization)
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        x_tensor = torch.from_numpy(x)
        x_tensor = x_tensor.to(torch.complex64) if np.iscomplexobj(x) else x_tensor.float()
        return x_tensor, torch.from_numpy(y).float()
