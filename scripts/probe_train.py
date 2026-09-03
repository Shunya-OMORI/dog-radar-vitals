"""config を 1 つ学習して run_dir を出すだけの最小ランナー。

プローブ (CLAUDE.md R2) 用。`run_comparison.py` は学習後に `evaluate_run` を呼ぶが、
heatmap モデルの下流指標 (F1 / RR間隔MAE / RMSSD誤差) はそこでは測れないので、
学習だけを担当し、評価は `evaluate_spatial_gnn_hrv.py` に任せる。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.train import make_run_dir, train_from_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    config = load_config(args.config)
    run_dir = make_run_dir(config, tag=args.tag)
    print(f"run_dir: {run_dir}", flush=True)
    train_from_config(config, run_dir)
    print(f"run_dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
