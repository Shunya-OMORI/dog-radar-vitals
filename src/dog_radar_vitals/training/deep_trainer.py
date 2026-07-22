"""深層モデル(PyTorch)の学習ループ。run_dirへconfig・環境・metrics・重みを記録する。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.dataset import WindowedVitalsDataset
from dog_radar_vitals.models.deep.registry import build_deep_model
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.schedulers import build_scheduler


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, float]:
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


def train_deep(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    task = config["task"]

    train_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["train"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    val_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["val"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
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

    n_bins = train_ds[0][0].shape[-1]
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_deep_model(config["model"]["name"], n_bins=n_bins, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    history = []
    best_val_mae = float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_metrics['loss']:.4f} "
            f"train_mae={train_metrics['mae']:.3f} val_loss={val_metrics['loss']:.4f} val_mae={val_metrics['mae']:.3f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_mae": best_val_mae}, indent=2))


def evaluate_deep(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = WindowedVitalsDataset(
        raw_root, data_cfg["dogs"]["test"], config["task"], data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    n_bins = test_ds[0][0].shape[-1]
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_deep_model(config["model"]["name"], n_bins=n_bins, **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    return _run_epoch(model, test_loader, device)
