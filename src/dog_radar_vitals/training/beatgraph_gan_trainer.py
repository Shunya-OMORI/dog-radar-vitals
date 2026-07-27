"""RCG beat segment -> PQRSTグラフ回帰(GNN)のGAN拡張。family=`mmecg_beatgraph_gan`。

`beatgraph_trainer.py`（MSE回帰のみ）に、`beatgraph_discriminator.py`との交互学習を追加する。
生成器(BeatGraphGNN)の損失はMSE再構成 + λ×敵対的損失(LSGAN)。判別器はレーダ入力に依存しない
条件なし判別器で、「PQRST波形として尤もらしい形」かどうかだけを見る（真のグラフ分布への
正則化として機能させる狙い）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_beatgraph_dataset import TIME_NORM_MS, MMECGBeatGraphDataset
from dog_radar_vitals.models.deep.beatgraph_discriminator import BeatGraphDiscriminator
from dog_radar_vitals.models.deep.beatgraph_gnn import BeatGraphGNN
from dog_radar_vitals.seeding import make_generator, set_all_seeds


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGBeatGraphDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGBeatGraphDataset(raw_root, trial_ids, segment_sec=data_cfg["segment_sec"])


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
    mse_loss = nn.MSELoss()

    total_recon, total_g_adv, total_d_loss, total_time_mae_ms, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = generator(x)

        # 判別器の更新: 真のグラフ->1, 生成グラフ(detach)->0
        d_optimizer.zero_grad()
        d_real = discriminator(y)
        d_fake = discriminator(pred.detach())
        d_loss = mse_loss(d_real, torch.ones_like(d_real)) + mse_loss(d_fake, torch.zeros_like(d_fake))
        d_loss.backward()
        d_optimizer.step()

        # 生成器の更新: 再構成MSE + adv_weight * 敵対的損失(判別器を騙す方向)
        g_optimizer.zero_grad()
        recon_loss = mse_loss(pred, y)
        d_fake_for_g = discriminator(pred)
        g_adv_loss = mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
        g_loss = recon_loss + adv_weight * g_adv_loss
        g_loss.backward()
        g_optimizer.step()

        with torch.no_grad():
            time_mae_ms = (pred[..., 0] - y[..., 0]).abs().mean().item() * TIME_NORM_MS

        total_recon += recon_loss.item()
        total_g_adv += g_adv_loss.item()
        total_d_loss += d_loss.item()
        total_time_mae_ms += time_mae_ms
        n_batches += 1

    return {
        "recon_loss": total_recon / n_batches,
        "g_adv_loss": total_g_adv / n_batches,
        "d_loss": total_d_loss / n_batches,
        "time_mae_ms": total_time_mae_ms / n_batches,
    }


def _run_eval_epoch(generator: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    generator.eval()
    mse_loss = nn.MSELoss()
    total_recon, total_time_mae_ms, total_amp_mae, n_batches = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = generator(x)
            recon_loss = mse_loss(pred, y)
            total_recon += recon_loss.item()
            total_time_mae_ms += (pred[..., 0] - y[..., 0]).abs().mean().item() * TIME_NORM_MS
            total_amp_mae += (pred[..., 1] - y[..., 1]).abs().mean().item()
            n_batches += 1
    return {
        "loss": total_recon / n_batches,
        "time_mae_ms": total_time_mae_ms / n_batches,
        "amp_mae": total_amp_mae / n_batches,
    }


def train_beatgraph_gan(config: dict, run_dir: Path, repo_root: Path) -> None:
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

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name", "adv_weight")}
    generator = BeatGraphGNN(**model_kwargs).to(device)
    discriminator = BeatGraphDiscriminator().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    adv_weight = config["model"].get("adv_weight", 0.1)

    history = []
    best_val_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        train_metrics = _run_train_epoch(generator, discriminator, train_loader, device, g_optimizer, d_optimizer, adv_weight)
        val_metrics = _run_eval_epoch(generator, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] recon={train_metrics['recon_loss']:.4f} d_loss={train_metrics['d_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_time_mae_ms={val_metrics['time_mae_ms']:.1f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(generator.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_loss": best_val_loss}, indent=2))


def evaluate_beatgraph_gan(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name", "adv_weight")}
    generator = BeatGraphGNN(**model_kwargs).to(device)
    generator.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_eval_epoch(generator, test_loader, device)
    return {"mae": metrics["time_mae_ms"], "loss": metrics["loss"], "amp_mae": metrics["amp_mae"]}
