"""Ahmed et al. (2024) UWB-DVS データセット シナリオ1（麻酔下のイヌ10頭）の読み込み。

各録音は Radar/RawData_No{n}.csv（9000行×467列、50 FPS、180秒）と、
BIONET/Reference_HR_No{n}.csv・Reference_BR_No{n}.csv（各180行、1 FPS）からなる。
参照値は拍単位ではなく1FPSへ平均化された心拍数・呼吸数のスカラ値である。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCENARIO_DIR_NAME = "Scenario1_FaintedDogs"
RADAR_FS_HZ = 50
REF_FS_HZ = 1


@dataclass(frozen=True)
class Recording:
    dog_id: str
    radar: np.ndarray  # shape (9000, 467), float32
    hr: np.ndarray  # shape (180,), float32, bpm
    br: np.ndarray  # shape (180,), float32, breaths/min


def list_dogs(raw_root: Path) -> list[str]:
    """data/raw/Scenario1_FaintedDogs 直下の No* ディレクトリ名を昇順で返す。"""
    scenario_dir = Path(raw_root) / SCENARIO_DIR_NAME
    dog_dirs = sorted(
        (p.name for p in scenario_dir.iterdir() if p.is_dir() and p.name.startswith("No")),
        key=lambda name: int(name.removeprefix("No")),
    )
    return dog_dirs


def load_recording(raw_root: Path, dog_id: str) -> Recording:
    dog_dir = Path(raw_root) / SCENARIO_DIR_NAME / dog_id
    radar = pd.read_csv(dog_dir / "Radar" / f"RawData_{dog_id}.csv", header=None).to_numpy(dtype=np.float32)
    hr = pd.read_csv(dog_dir / "BIONET" / f"Reference_HR_{dog_id}.csv", header=None).to_numpy(dtype=np.float32).squeeze(-1)
    br = pd.read_csv(dog_dir / "BIONET" / f"Reference_BR_{dog_id}.csv", header=None).to_numpy(dtype=np.float32).squeeze(-1)
    return Recording(dog_id=dog_id, radar=radar, hr=hr, br=br)
