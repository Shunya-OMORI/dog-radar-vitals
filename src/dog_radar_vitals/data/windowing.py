"""録音1本をスライディング窓に切り出す共通ロジック。

深層モデル（生の窓をそのまま入力する `dataset.py`）と古典MLモデル（窓から
手作り特徴量を抽出する `features.py`）の両方がここを経由することで、
「どの時刻の窓がどの参照値に対応するか」という整合性を一箇所に保つ。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np

from dog_radar_vitals.data.scenario1 import RADAR_FS_HZ, Recording

Task = Literal["hr", "br"]


def zscore(radar: np.ndarray) -> np.ndarray:
    mean = radar.mean(axis=0, keepdims=True)
    std = radar.std(axis=0, keepdims=True) + 1e-8
    return (radar - mean) / std


def iter_windows(
    rec: Recording, task: Task, window_sec: int, stride_sec: int
) -> Iterator[tuple[np.ndarray, float]]:
    """(正規化済みレーダ窓, 窓終端秒の参照値) を順に返す。

    窓終端に対応する参照値（1Hz）のインデックスは `end // RADAR_FS_HZ - 1`。
    例えばwindow_sec=10で最初の窓が [0, 500) なら、9秒目（0始まりでidx=9）の値を使う。
    """
    radar = zscore(rec.radar)
    target_series = rec.hr if task == "hr" else rec.br

    window_len = window_sec * RADAR_FS_HZ
    stride = stride_sec * RADAR_FS_HZ

    n_steps = radar.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        ref_idx = end // RADAR_FS_HZ - 1
        if ref_idx < 0 or ref_idx >= len(target_series):
            continue
        yield radar[start:end], float(target_series[ref_idx])
