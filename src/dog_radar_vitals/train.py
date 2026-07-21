"""設定ファイル(YAML)を受け取り、1つのモデルを学習してrunsディレクトリに記録する。

runs/{run_id}/ に、後から追跡できるよう以下を残す:
  config.yaml    - 実際に使われた設定のスナップショット（extends解決後）
  metrics.json   - epochごとのtrain/val損失とMAE
  best_model.pt  - val MAE最良時点のモデル重み
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.config import load_config
from dog_radar_vitals.data.dataset import WindowedVitalsDataset
from dog_radar_vitals.models.registry import build_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_abs_err, n = 0.0, 0.0, 0
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(-1)
            pred = model(x)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_abs_err += (pred - y).abs().sum().item()
            n += batch_size

    return {"loss": total_loss / n, "mae": total_abs_err / n}


def train(config: dict, run_dir: Path) -> None:
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = REPO_ROOT / data_cfg["raw_root"]
    task = config["task"]

    train_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["train"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    val_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["val"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
    )

    train_cfg = config["train"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    n_bins = train_ds[0][0].shape[-1]
    model = build_model(n_bins=n_bins, **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    history = []
    best_val_mae = float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_mae={train_metrics['mae']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.3f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_mae": best_val_mae}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/experiments/*.yaml へのパス")
    args = parser.parse_args()

    config = load_config(args.config)

    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{config['task']}_{config['model']['name']}"
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))

    print(f"run_dir: {run_dir}")
    train(config, run_dir)


if __name__ == "__main__":
    main()
