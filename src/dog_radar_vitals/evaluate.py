"""学習済みrunをtest split（犬単位で学習に未使用の個体）で評価するCLI。

`train.py` と同様に `config["model"]["family"]` で deep/classical を振り分ける。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dog_radar_vitals.config import load_config
from dog_radar_vitals.train import REPO_ROOT
from dog_radar_vitals.training.classical_trainer import evaluate_classical
from dog_radar_vitals.training.deep_trainer import evaluate_deep

_EVAL_FNS = {"deep": evaluate_deep, "classical": evaluate_classical}


def evaluate_run(run_dir: Path) -> dict[str, float]:
    config = load_config(run_dir / "config.yaml")
    family = config["model"]["family"]
    if family not in _EVAL_FNS:
        raise ValueError(f"unknown model family '{family}'. expected one of {sorted(_EVAL_FNS)}")

    test_metrics = _EVAL_FNS[family](run_dir, config, REPO_ROOT)
    print(f"test_mae={test_metrics['mae']:.3f}")

    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="runs/{run_id} へのパス")
    args = parser.parse_args()
    evaluate_run(Path(args.run))


if __name__ == "__main__":
    main()
