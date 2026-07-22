"""設定ファイル(YAML)を受け取り、1つのモデルを学習してrunsディレクトリに記録するCLI。

`config["model"]["family"]` で処理を振り分ける: "deep"->`training/deep_trainer.py`
（犬HR/BR、窓->スカラ回帰）、"classical"->`training/classical_trainer.py`（同、scikit-learn）、
"ecg_seq2seq"->`training/ecg_trainer.py`（ヒトレーダI/Q->ECG波形、窓->波形回帰）。
いずれも runs/{run_id}/ に config.yaml・environment.json・metrics.json・
モデル重みを残す（詳細は runs/README.md）。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from dog_radar_vitals.config import load_config
from dog_radar_vitals.reproducibility import capture_environment
from dog_radar_vitals.training.classical_trainer import train_classical
from dog_radar_vitals.training.deep_trainer import train_deep
from dog_radar_vitals.training.ecg_trainer import train_ecg
from dog_radar_vitals.training.rpeak_trainer import train_rpeak

REPO_ROOT = Path(__file__).resolve().parents[2]

_TRAIN_FNS = {
    "deep": train_deep,
    "classical": train_classical,
    "ecg_seq2seq": train_ecg,
    "rpeak_seq2seq": train_rpeak,
}


def make_run_dir(config: dict, tag: str | None = None) -> Path:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{config['task']}_{config['model']['name']}"
    if tag:
        run_id = f"{run_id}_{tag}"
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train_from_config(config: dict, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    (run_dir / "environment.json").write_text(json.dumps(capture_environment(REPO_ROOT), indent=2))

    family = config["model"]["family"]
    if family not in _TRAIN_FNS:
        raise ValueError(f"unknown model family '{family}'. expected one of {sorted(_TRAIN_FNS)}")
    _TRAIN_FNS[family](config, run_dir, REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/experiments/*.yaml へのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = make_run_dir(config)
    print(f"run_dir: {run_dir}")
    train_from_config(config, run_dir)


if __name__ == "__main__":
    main()
