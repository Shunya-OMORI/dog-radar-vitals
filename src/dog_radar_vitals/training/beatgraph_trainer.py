"""RCG beat segment -> PQRSTグラフ回帰(GNN)の学習ループ。family=`mmecg_beatgraph`。

評価指標は正規化済み時刻オフセット・振幅のMSE損失に加え、解釈しやすいように
時刻オフセットMAE[ms]・振幅MAEも記録する（`data/mmecg_beatgraph_dataset.py`の
`TIME_NORM_MS`で正規化を戻す）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_beatgraph_dataset import TIME_NORM_MS, MMECGBeatGraphDataset
from dog_radar_vitals.models.deep.beatgraph_gnn import BeatGraphGNN
from dog_radar_vitals.seeding import make_generator, set_all_seeds


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_time_mae_ms, total_amp_mae, n_batches = 0.0, 0.0, 0.0, 0
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                time_mae_ms = (pred[..., 0] - y[..., 0]).abs().mean().item() * TIME_NORM_MS
                amp_mae = (pred[..., 1] - y[..., 1]).abs().mean().item()

            total_loss += loss.item()
            total_time_mae_ms += time_mae_ms
            total_amp_mae += amp_mae
            n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "time_mae_ms": total_time_mae_ms / n_batches,
        "amp_mae": total_amp_mae / n_batches,
    }


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGBeatGraphDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGBeatGraphDataset(raw_root, trial_ids, segment_sec=data_cfg["segment_sec"])


def train_beatgraph(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    train_ds = _build_dataset(data_cfg, raw_root, "train")
    val_ds = _build_dataset(data_cfg, raw_root, "val")

    train_cfg = config["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        generator=make_generator(config["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = BeatGraphGNN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    history = []
    best_val_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_time_mae_ms={train_metrics['time_mae_ms']:.1f} "
            f"val_loss={val_metrics['loss']:.4f} val_time_mae_ms={val_metrics['time_mae_ms']:.1f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_loss": best_val_loss}, indent=2))


def evaluate_beatgraph(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = BeatGraphGNN(**model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    return {"mae": metrics["time_mae_ms"], "loss": metrics["loss"], "amp_mae": metrics["amp_mae"]}
