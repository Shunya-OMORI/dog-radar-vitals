"""全91トライアルのSST(50ch)を事前計算し`data/processed/mmecg_sst/`にキャッシュする。

1トライアルあたり約30秒(50ch)かかるため、`multiprocessing`でトライアル単位に並列化する。

使い方:
    python scripts/precompute_mmecg_sst.py
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.data.mmecg import list_trials  # noqa: E402
from dog_radar_vitals.data.mmecg_sst_cache import get_or_compute_sst  # noqa: E402

RAW_ROOT = REPO_ROOT / "data" / "raw"


def _compute_one(trial_id: int) -> tuple[int, float]:
    t0 = time.time()
    get_or_compute_sst(RAW_ROOT, trial_id)
    return trial_id, time.time() - t0


def main() -> None:
    trial_ids = list_trials(RAW_ROOT)
    print(f"{len(trial_ids)} trials to process")
    with Pool(processes=8) as pool:
        for trial_id, elapsed in pool.imap_unordered(_compute_one, trial_ids):
            print(f"trial {trial_id}: {elapsed:.1f}s")
    print("done")


if __name__ == "__main__":
    main()
