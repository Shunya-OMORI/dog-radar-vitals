"""学習済みrunをtest split（犬単位で学習に未使用の個体）で評価するCLI。

`train.py` と同様に `config["model"]["family"]` で deep/classical を振り分ける。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dog_radar_vitals.config import load_config
from dog_radar_vitals.train import REPO_ROOT
from dog_radar_vitals.training.beatgraph_gan_trainer import evaluate_beatgraph_gan
from dog_radar_vitals.training.beatgraph_trainer import evaluate_beatgraph
from dog_radar_vitals.training.chen2022_trainer import evaluate_chen2022
from dog_radar_vitals.training.classical_trainer import evaluate_classical
from dog_radar_vitals.training.radarode_longterm_trainer import evaluate_radarode_longterm
from dog_radar_vitals.training.radarode_sceg_trainer import evaluate_radarode_sceg
from dog_radar_vitals.training.singlecycle_cnn_trainer import evaluate_singlecycle_cnn
from dog_radar_vitals.training.tau_predictor_trainer import evaluate_tau_predictor
from dog_radar_vitals.training.deep_trainer import evaluate_deep
from dog_radar_vitals.training.ecg_trainer import evaluate_ecg
from dog_radar_vitals.training.heatmap_gan_trainer import evaluate_heatmap_gan
from dog_radar_vitals.training.mmecg_classical_trainer import evaluate_mmecg_classical
from dog_radar_vitals.training.mmecg_rpeak_trainer import evaluate_mmecg_rpeak
from dog_radar_vitals.training.mmecg_trainer import evaluate_mmecg
from dog_radar_vitals.training.rpeak_trainer import evaluate_rpeak
from dog_radar_vitals.training.spatial_fusion_trainer import evaluate_spatial_fusion
from dog_radar_vitals.training.spatial_heatmap_gan_trainer import evaluate_spatial_heatmap_gan

_EVAL_FNS = {
    "deep": evaluate_deep,
    "classical": evaluate_classical,
    "ecg_seq2seq": evaluate_ecg,
    "rpeak_seq2seq": evaluate_rpeak,
    "mmecg_seq2seq": evaluate_mmecg,
    "mmecg_rpeak_seq2seq": evaluate_mmecg_rpeak,
    "mmecg_classical_rr": evaluate_mmecg_classical,
    "mmecg_beatgraph": evaluate_beatgraph,
    "mmecg_beatgraph_gan": evaluate_beatgraph_gan,
    "mmecg_heatmap_gan": evaluate_heatmap_gan,
    "mmecg_spatial_seq2seq": evaluate_spatial_fusion,
    "mmecg_spatial_heatmap_gan": evaluate_spatial_heatmap_gan,
    "mmecg_chen2022": evaluate_chen2022,
    "mmecg_radarode_sceg": evaluate_radarode_sceg,
    "mmecg_singlecycle_cnn": evaluate_singlecycle_cnn,
    "mmecg_tau_predictor": evaluate_tau_predictor,
    "mmecg_radarode_longterm": evaluate_radarode_longterm,
}


def evaluate_run(run_dir: Path) -> dict[str, float]:
    config = load_config(run_dir / "config.yaml")
    family = config["model"]["family"]
    if family not in _EVAL_FNS:
        raise ValueError(f"unknown model family '{family}'. expected one of {sorted(_EVAL_FNS)}")

    test_metrics = _EVAL_FNS[family](run_dir, config, REPO_ROOT)
    # familyによって主要指標のキーが異なる（deep/classicalはmae、ecg_seq2seqはcorr）ので存在するものを出す。
    primary_key = "mae" if "mae" in test_metrics else "corr"
    print(f"test_{primary_key}={test_metrics[primary_key]:.3f}")

    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="runs/{run_id} へのパス")
    args = parser.parse_args()
    evaluate_run(Path(args.run))


if __name__ == "__main__":
    main()
