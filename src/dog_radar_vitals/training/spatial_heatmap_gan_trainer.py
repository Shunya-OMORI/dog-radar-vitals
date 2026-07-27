"""RCG+posXYZ空間融合/GNN生成器 -> R波heatmapの敵対的リファインメント。family=`mmecg_spatial_heatmap_gan`。

`heatmap_gan_trainer.py`（生成器は1引数`forward(x)`のecg_registryモデル）と同じLSGAN交互学習
パターンだが、生成器が`rpeak_spatial_fusion`/`rpeak_spatial_gnn`のような2引数
`forward(rcg, posxyz)`を取る点が異なるため別ファイルとする。ユーザから「空間座標をGANにも
使ってみては」との提案を受け、これまで試した中で最高のF1(検出率)を達成した空間融合系
生成器に、GANによる敵対的リファインメントが追加の改善をもたらすかを検証する。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset
from dog_radar_vitals.models.deep.heatmap_patch_discriminator import HeatmapPatchDiscriminator
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.spatial_fusion_trainer import build_spatial_model


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGSpatialWindowDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSpatialWindowDataset(raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"], heatmap=True)


def _build_generator(model_cfg: dict) -> nn.Module:
    return build_spatial_model(model_cfg)


def _run_train_epoch(
    generator: nn.Module,
    discriminator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    adv_weight: float,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    total_recon, total_g_adv, total_d_loss, total_corr, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    for rcg, posxyz, y in loader:
        rcg, posxyz, y = rcg.to(device), posxyz.to(device), y.to(device)

        pred = generator(rcg, posxyz)

        d_optimizer.zero_grad()
        d_real = discriminator(y, rcg)
        d_fake = discriminator(pred.detach(), rcg)
        d_loss = mse_loss(d_real, torch.ones_like(d_real)) + mse_loss(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        recon_loss = bce_loss(pred, y)
        d_fake_for_g = discriminator(pred, rcg)
        g_adv_loss = mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
        g_loss = recon_loss + adv_weight * g_adv_loss
        g_loss.backward()
        g_optimizer.step()

        total_recon += recon_loss.item()
        total_g_adv += g_adv_loss.item()
        total_d_loss += d_loss.item()
        total_corr += _pearson_corr(pred.detach(), y)
        n_batches += 1

    return {
        "recon_loss": total_recon / n_batches,
        "g_adv_loss": total_g_adv / n_batches,
        "d_loss": total_d_loss / n_batches,
        "corr": total_corr / n_batches,
    }


def _run_eval_epoch(generator: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    generator.eval()
    bce_loss = nn.BCELoss()
    total_loss, total_corr, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for rcg, posxyz, y in loader:
            rcg, posxyz, y = rcg.to(device), posxyz.to(device), y.to(device)
            pred = generator(rcg, posxyz)
            total_loss += bce_loss(pred, y).item()
            total_corr += _pearson_corr(pred, y)
            n_batches += 1
    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def train_spatial_heatmap_gan(config: dict, run_dir: Path, repo_root: Path) -> None:
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

    generator = _build_generator(config["model"]).to(device)
    disc_kwargs = config["model"].get("discriminator", {})
    discriminator = HeatmapPatchDiscriminator(condition_channels=config["model"]["n_points"], **disc_kwargs).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    adv_weight = config["model"].get("adv_weight", 0.1)

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_metrics = _run_train_epoch(generator, discriminator, train_loader, device, g_optimizer, d_optimizer, adv_weight)
        val_metrics = _run_eval_epoch(generator, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] recon={train_metrics['recon_loss']:.4f} d_loss={train_metrics['d_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(generator.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_spatial_heatmap_gan(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    generator = _build_generator(config["model"]).to(device)
    generator.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_eval_epoch(generator, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
