"""radarODEの長期ECG再構成ネットワークの学習ループ。family=`mmecg_radarode_longterm`。

`config["model"]["sceg_run_dir"]`で指定した学習済み(凍結)SCEGを使い、
`MMECGRadarODELongtermDataset`でmorphological referenceを都度生成しながら学習する。
評価指標は既存の`ecg_*`系と同じ窓内Pearson相関（`201`等と直接比較できるように）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.config import load_config
from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_radarode_longterm_dataset import MMECGRadarODELongtermDataset
from dog_radar_vitals.models.deep.radarode_longterm import RadarODELongTerm
from dog_radar_vitals.models.deep.radarode_sceg import RadarODESCEG
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.schedulers import build_scheduler


def _load_frozen_sceg(sceg_run_dir: Path, repo_root: Path, device: torch.device) -> RadarODESCEG:
    sceg_config = load_config(repo_root / sceg_run_dir / "config.yaml")
    model_kwargs = {k: v for k, v in sceg_config["model"].items() if k not in ("family", "name")}
    model = RadarODESCEG(**model_kwargs).to(device)
    model.load_state_dict(torch.load(repo_root / sceg_run_dir / "best_model.pt", map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _build_dataset(
    data_cfg: dict, raw_root: Path, split: str, sceg_model: RadarODESCEG, device: torch.device
) -> MMECGRadarODELongtermDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGRadarODELongtermDataset(
        raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"], sceg_model, device
    )


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, n_batches = 0.0, 0.0, 0
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for rcg, ref, ecg in loader:
            rcg, ref, ecg = rcg.to(device), ref.to(device), ecg.to(device)
            pred = model(rcg, ref)
            loss = loss_fn(pred, ecg)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), ecg)
            n_batches += 1

    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def train_radarode_longterm(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    sceg_model = _load_frozen_sceg(Path(config["model"]["sceg_run_dir"]), repo_root, device)

    train_ds = _build_dataset(data_cfg, raw_root, "train", sceg_model, device)
    val_ds = _build_dataset(data_cfg, raw_root, "val", sceg_model, device)
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    train_cfg = config["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=make_generator(config["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name", "sceg_run_dir")}
    model = RadarODELongTerm(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
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


def evaluate_radarode_longterm(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    sceg_model = _load_frozen_sceg(Path(config["model"]["sceg_run_dir"]), repo_root, device)

    test_ds = _build_dataset(data_cfg, raw_root, "test", sceg_model, device)
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name", "sceg_run_dir")}
    model = RadarODELongTerm(**model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
