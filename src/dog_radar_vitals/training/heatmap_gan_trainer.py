"""RCG(mmWave) -> R波heatmapの敵対的リファインメント学習。family=`mmecg_heatmap_gan`。

生成器は`ecg_registry.py`の既存heatmapモデル（既定`rpeak_unet1d`、フェーズDの成果を流用）を
そのまま使い、`heatmap_patch_discriminator.HeatmapPatchDiscriminator`との交互学習
（LSGAN損失）を追加する。生成器損失=BCE再構成 + adv_weight×敵対的損失
（`training/beatgraph_gan_trainer.py`と同じLSGAN交互学習パターンを踏襲）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_rpeak_dataset import MMECGRPeakWindowDataset
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model
from dog_radar_vitals.models.deep.heatmap_patch_discriminator import HeatmapPatchDiscriminator
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGRPeakWindowDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGRPeakWindowDataset(
        raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"], complex_input=data_cfg.get("complex_input", False)
    )


_NON_GENERATOR_KEYS = ("family", "name", "discriminator", "adv_weight")


def _build_generator(model_cfg: dict) -> nn.Module:
    """model configをフラットに書く（他familyと同じ規約）。生成器以外のキー
    (`discriminator`のネスト辞書・`adv_weight`)だけ除外してbuild_ecg_modelに渡す。
    `mmecg_rpeak_evaluation.py`が同じフィルタ規約で読めるよう`family`/`name`のみ除外する
    既存パターンをここでも踏襲する（差分は discriminator/adv_weight の2キー）。
    """
    gen_kwargs = {k: v for k, v in model_cfg.items() if k not in _NON_GENERATOR_KEYS}
    return build_ecg_model(model_cfg["name"], **gen_kwargs)


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
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = generator(x)

        d_optimizer.zero_grad()
        d_real = discriminator(y, x)
        d_fake = discriminator(pred.detach(), x)
        d_loss = mse_loss(d_real, torch.ones_like(d_real)) + mse_loss(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        recon_loss = bce_loss(pred, y)
        d_fake_for_g = discriminator(pred, x)
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = generator(x)
            total_loss += bce_loss(pred, y).item()
            total_corr += _pearson_corr(pred, y)
            n_batches += 1
    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def train_heatmap_gan(config: dict, run_dir: Path, repo_root: Path) -> None:
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
    discriminator = HeatmapPatchDiscriminator(condition_channels=config["model"]["in_channels"], **disc_kwargs).to(device)

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


def evaluate_heatmap_gan(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    generator = _build_generator(config["model"]).to(device)
    generator.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_eval_epoch(generator, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
