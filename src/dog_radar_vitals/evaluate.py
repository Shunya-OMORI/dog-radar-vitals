"""学習済みrunをtest split（犬単位で学習に未使用の個体）で評価する。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dog_radar_vitals.config import load_config
from dog_radar_vitals.data.dataset import WindowedVitalsDataset
from dog_radar_vitals.models.registry import build_model
from dog_radar_vitals.train import REPO_ROOT, run_epoch


def evaluate_run(run_dir: Path) -> dict[str, float]:
    config = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = REPO_ROOT / data_cfg["raw_root"]
    test_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["test"], config["task"], data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    n_bins = test_ds[0][0].shape[-1]
    model = build_model(n_bins=n_bins, **config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    test_metrics = run_epoch(model, test_loader, device)
    print(f"test_loss={test_metrics['loss']:.4f} test_mae={test_metrics['mae']:.3f}")

    (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="runs/{run_id} へのパス")
    args = parser.parse_args()
    evaluate_run(Path(args.run))


if __name__ == "__main__":
    main()
