"""RCG(50ch, MMECG) -> ECG波形（sequence-to-sequence）の学習ループ。

`ecg_trainer.py`（Schellenberger、2ch I/Q）とほぼ同一だが、(a) 入力データセットが
`MMECGWindowDataset`である点、(b) `data.subjects`が被験者ID(int)のリストであり、
トライアルID(.matファイル)への変換を`trial_ids_for_subjects`で行う点が異なる
（91トライアルが11被験者に集約されるため、被験者単位でsplitを切る必要がある。
`data/raw/README.md`参照）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_dataset import MMECGWindowDataset
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.schedulers import build_scheduler


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None,
               amplitude_weight_beta: float = 0.0) -> dict[str, float]:
    """amplitude_weight_beta=0.0 (default): unweighted MSE, unchanged behavior.
    >0: per-timestep weight = 1 + beta*|y| (y is z-scored ECG), so QRS-region
    timesteps (large |y|) get more weight than the mostly-flat baseline that
    otherwise dominates plain per-timestep MSE over an 800-sample window
    (2026-08-05, testing whether loss design -- not just regularization
    strength -- explains the train/val corr gap seen in every prior MMECG
    ecg_cnn1d/ecg_conv_ncp run)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, n_batches = 0.0, 0.0, 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if amplitude_weight_beta > 0:
                weight = 1.0 + amplitude_weight_beta * y.abs()
                loss = (weight * (pred - y) ** 2).mean()
            else:
                loss = nn.functional.mse_loss(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), y)
            n_batches += 1

    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGWindowDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGWindowDataset(
        raw_root,
        trial_ids,
        data_cfg["window_sec"],
        data_cfg["stride_sec"],
        complex_input=data_cfg.get("complex_input", False),
        normalization=data_cfg.get("normalization", "zscore"),
    )


def train_mmecg(config: dict, run_dir: Path, repo_root: Path) -> None:
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
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    amplitude_weight_beta = float(train_cfg.get("amplitude_weight_beta", 0.0))

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_metrics = _run_epoch(model, train_loader, device, optimizer, amplitude_weight_beta)
        val_metrics = _run_epoch(model, val_loader, device, amplitude_weight_beta=amplitude_weight_beta)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_metrics['loss']:.4f} "
            f"train_corr={train_metrics['corr']:.3f} val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_mmecg(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
