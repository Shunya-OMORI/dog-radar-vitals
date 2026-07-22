"""Schellenberger et al. (2020) CW radarデータセットの読み込み。

ヒト30名の安静・自律神経賦活シナリオを、24GHz CW radarのI/Q信号と
臨床用心電図(ECG)を同期記録している。犬データセット（1Hz平均値のみ）と異なり、
**レーダとECGが同一サンプリングレート(fs_radar=fs_ecg=2000Hz)・同一長で記録されている**ため、
拍単位・波形単位の対応が直接取れる。DOI: 10.1038/s41597-020-00629-5, Figshare 12186516。

「イヌのRR Interval・ECG波形推定」という卒論の本題にはこのデータセット自体は使えない
（対象がヒトのため）が、レーダ→ECG波形推定という手法をヒトデータで先に検証し、
将来イヌの拍単位データが手に入った際に転用することを目的とする
（[`EXPERIMENTS.md`](../../../EXPERIMENTS.md) 参照）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

SCENARIO_DIR_NAME = "schellenberger_human"


@dataclass(frozen=True)
class HumanRecording:
    subject_id: str
    scenario: str
    fs: int  # レーダ・ECGとも同一サンプリングレート[Hz]
    radar_i: np.ndarray  # shape (T,), float32
    radar_q: np.ndarray  # shape (T,), float32
    ecg: np.ndarray  # shape (T,), float32 (tfm_ecg1)


def list_subjects(raw_root: Path) -> list[str]:
    scenario_dir = Path(raw_root) / SCENARIO_DIR_NAME
    return sorted(p.name for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("GDN"))


def load_recording(raw_root: Path, subject_id: str, scenario: str = "Resting") -> HumanRecording:
    subject_dir = Path(raw_root) / SCENARIO_DIR_NAME / subject_id
    matches = list(subject_dir.glob(f"{subject_id}_*_{scenario}.mat"))
    if not matches:
        raise FileNotFoundError(f"{subject_dir} に scenario='{scenario}' の.matファイルが見つからない")

    d = sio.loadmat(matches[0])
    fs_radar = int(d["fs_radar"].item())
    fs_ecg = int(d["fs_ecg"].item())
    if fs_radar != fs_ecg:
        raise ValueError(f"{matches[0]}: fs_radar({fs_radar}) != fs_ecg({fs_ecg})、想定外のファイル")

    return HumanRecording(
        subject_id=subject_id,
        scenario=scenario,
        fs=fs_radar,
        radar_i=d["radar_i"].astype(np.float32).squeeze(-1),
        radar_q=d["radar_q"].astype(np.float32).squeeze(-1),
        ecg=d["tfm_ecg1"].astype(np.float32).squeeze(-1),
    )
