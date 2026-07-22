"""レーダI/Q -> R波heatmap（疎なイベント検出）の学習ループ。

`ecg_trainer.py`（密な波形回帰、MSE損失・相関係数評価）とは損失関数が異なる
（heatmapは[0,1]の疑似確率としてBCE損失を使う）。学習中の指標は`ecg_trainer.py`と
同じ形式（相関係数）で揃えておき、`evaluate.py`の汎用dispatchがそのまま使えるようにする。
RR Interval自体の精度評価は `scripts/compare_ecg_vs_rpeak.py` で別途行う。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.rpeak_dataset import RPeakWindowDataset
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, n_batches = 0.0, 0.0, 0
    loss_fn = nn.BCELoss()

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), y)
            n_batches += 1

    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def train_rpeak(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    scenario = data_cfg["scenario"]

    train_ds = RPeakWindowDataset(
        raw_root, data_cfg["subjects"]["train"], scenario, data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    val_ds = RPeakWindowDataset(
        raw_root, data_cfg["subjects"]["val"], scenario, data_cfg["window_sec"], data_cfg["stride_sec"]
    )

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
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_corr={train_metrics['corr']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_rpeak(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = RPeakWindowDataset(
        raw_root, data_cfg["subjects"]["test"], data_cfg["scenario"], data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
