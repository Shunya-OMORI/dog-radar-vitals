"""レーダ時系列をスライディング窓に切り出し、窓終端時刻の心拍数/呼吸数を目的変数とするDataset。

犬ごとに正規化統計量が変わりうるため、正規化は録音（犬）単位で行う。
train/val/testの分割は「犬ID単位」で行い、同一犬のウィンドウが複数splitに跨って
リークすることを防ぐ。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from dog_radar_vitals.data.scenario1 import RADAR_FS_HZ, Recording, load_recording

Task = Literal["hr", "br"]


def _zscore(radar: np.ndarray) -> np.ndarray:
    mean = radar.mean(axis=0, keepdims=True)
    std = radar.std(axis=0, keepdims=True) + 1e-8
    return (radar - mean) / std


class WindowedVitalsDataset(Dataset):
    """1つの録音を、`window_sec`秒幅・`stride_sec`秒刻みの窓に切り出したデータセット。

    各サンプルは窓終端の1秒に対応する参照値（HRまたはBR）を目的変数とする。
    """

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
            self._samples.extend(self._make_windows(rec))

    def _make_windows(self, rec: Recording) -> list[tuple[np.ndarray, float]]:
        radar = _zscore(rec.radar)
        target_series = rec.hr if self.task == "hr" else rec.br

        window_len = self.window_sec * RADAR_FS_HZ
        stride = self.stride_sec * RADAR_FS_HZ

        windows = []
        n_steps = radar.shape[0]
        for start in range(0, n_steps - window_len + 1, stride):
            end = start + window_len
            # 窓終端に対応する参照値（REF_FS_HZ=1Hzなのでend//RADAR_FS_HZ秒目）のインデックス
            ref_idx = end // RADAR_FS_HZ - 1
            if ref_idx < 0 or ref_idx >= len(target_series):
                continue
            windows.append((radar[start:end], float(target_series[ref_idx])))
        return windows

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._samples[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.float32)
