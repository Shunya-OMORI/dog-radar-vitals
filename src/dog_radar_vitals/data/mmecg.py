"""MMECG（Chen et al. 2022、mmWave radar）データセットの読み込み。

Schellenbergerデータセット（CW radar、ヒト30名）との違いは、(a) レーダ入力が2ch(I/Q)ではなく
50ch（RCG: 50点の3D心臓表面変位計測）である点、(b) 91トライアルが11被験者に集約される
（1被験者あたり2〜23トライアル）ため、トライアル単位ではなく被験者ID単位でsplitを切らないと
個体リークが起きる点（`data/raw/README.md`参照）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

SCENARIO_DIR_NAME = "mmecg"
TRIAL_SUBDIR = "finalPartialPublicData20221108"
FS_HZ = 200


@dataclass(frozen=True)
class MMECGRecording:
    trial_id: int
    subject_id: int
    fs: int
    rcg: np.ndarray  # shape (T, 50), float32
    ecg: np.ndarray  # shape (T,), float32
    posxyz: np.ndarray  # shape (50, 3), float32。RCGの各点の3D座標。トライアルごとに異なる
    physistatus: str
    age: int
    gender: str


def _trial_dir(raw_root: Path) -> Path:
    return Path(raw_root) / SCENARIO_DIR_NAME / TRIAL_SUBDIR


def list_trials(raw_root: Path) -> list[int]:
    """展開済みの全トライアルID(.matのファイル名部分)を昇順で返す。"""
    return sorted(int(p.stem) for p in _trial_dir(raw_root).glob("*.mat"))


def load_trial(raw_root: Path, trial_id: int) -> MMECGRecording:
    path = _trial_dir(raw_root) / f"{trial_id}.mat"
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    d = m["data"]
    return MMECGRecording(
        trial_id=trial_id,
        subject_id=int(d.id),
        fs=FS_HZ,
        rcg=np.asarray(d.RCG, dtype=np.float32),
        ecg=np.asarray(d.ECG, dtype=np.float32),
        posxyz=np.asarray(d.posXYZ, dtype=np.float32),
        physistatus=str(d.physistatus),
        age=int(d.age),
        gender=str(d.gender),
    )


def trials_by_subject(raw_root: Path) -> dict[int, list[int]]:
    """被験者ID -> その被験者に属するトライアルID一覧。train/val/test split構築用。

    matファイルを毎回読み込むとコストが高いため、トライアルIDと被験者IDの対応をキャッシュせず
    都度読み込む単純な実装にしている（91ファイル程度なら数秒で終わる）。
    """
    mapping: dict[int, list[int]] = {}
    for trial_id in list_trials(raw_root):
        subject_id = load_trial(raw_root, trial_id).subject_id
        mapping.setdefault(subject_id, []).append(trial_id)
    return mapping


def trial_ids_for_subjects(raw_root: Path, subject_ids: list[int]) -> list[int]:
    """指定した被験者ID群に属する全トライアルIDを返す（train/val/test split構築の実用ヘルパ）。"""
    by_subject = trials_by_subject(raw_root)
    ids: list[int] = []
    for subject_id in subject_ids:
        ids.extend(by_subject.get(subject_id, []))
    return sorted(ids)
