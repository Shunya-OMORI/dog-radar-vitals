"""Chen et al. (2022, RCG2ECG)アーキテクチャ再現モデルの学習ループ。family=`mmecg_chen2022`。

`spatial_fusion_trainer.py`と同じくRCG+posXYZを入力に取るが、(a) 損失がMSEではなく
μ-law companding後の256値カテゴリカル分布に対するcross-entropy、(b) 訓練は教師強制
（正解ECGを1サンプル右シフトして入力）で並列に行うが、評価は原著と同じ自己回帰生成
（1サンプルずつ、前ステップの生成値を次の入力に使う）で行う点が異なる。

自己回帰生成は時系列長T回のforward呼び出しを要するため、CNN/Transformer系のfeedforward
回帰モデルに比べ評価が大幅に遅い（1バッチあたり概ねT回のGPU forward）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset
from dog_radar_vitals.data.mulaw import mu_law_decode, mu_law_encode, quantize
from dog_radar_vitals.models.deep.chen2022_reconstructor import Chen2022Reconstructor
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.schedulers import build_scheduler


def _build_model(model_cfg: dict) -> Chen2022Reconstructor:
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("family", "name")}
    return Chen2022Reconstructor(**kwargs)


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGSpatialWindowDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSpatialWindowDataset(
        raw_root,
        trial_ids,
        data_cfg["window_sec"],
        data_cfg["stride_sec"],
        heatmap=False,
        normalization=data_cfg.get("normalization", "minmax"),
    )


def _shift_right(companded: torch.Tensor) -> torch.Tensor:
    """companded ECG(batch, T)を1サンプル右シフトし(batch, 1, T)にする（教師強制の入力、先頭は0）。"""
    prev = torch.zeros_like(companded)
    prev[:, 1:] = companded[:, :-1]
    return prev.unsqueeze(1)


def _train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, loss_fn) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for rcg, posxyz, ecg in loader:
        rcg, posxyz, ecg = rcg.to(device), posxyz.to(device), ecg.to(device)
        companded = mu_law_encode(ecg)
        labels = quantize(companded)
        prev = _shift_right(companded)

        logits = model.forward_teacher_forced(rcg, posxyz, prev)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def _val_epoch_teacher_forced(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn) -> float:
    """検証は教師強制cross-entropyのみ（自己回帰生成は毎epoch行うには重すぎるため、
    epochごとのモデル選択には教師強制lossを使い、自己回帰評価はテスト時のみ行う）。
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    for rcg, posxyz, ecg in loader:
        rcg, posxyz, ecg = rcg.to(device), posxyz.to(device), ecg.to(device)
        companded = mu_law_encode(ecg)
        labels = quantize(companded)
        prev = _shift_right(companded)
        logits = model.forward_teacher_forced(rcg, posxyz, prev)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def train_chen2022(config: dict, run_dir: Path, repo_root: Path) -> None:
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

    model = _build_model(config["model"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    history = []
    best_val_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_loss = _train_epoch(model, train_loader, device, optimizer, loss_fn)
        val_loss = _val_epoch_teacher_forced(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss, "val_loss": val_loss})
        print(f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.2e} train_ce={train_loss:.4f} val_ce={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_loss": best_val_loss}, indent=2))


@torch.no_grad()
def evaluate_chen2022(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    """自己回帰生成でtest setを評価し、窓内相関(既存指標と同じ定義)を返す。

    拍単位相関は`scripts/evaluate_per_beat_correlation.py`側で別途計算する
    （本評価関数はそちらと同じ`evaluate_ecg`系のインタフェース(corr, loss)に揃える）。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model = _build_model(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    total_corr, n_batches = 0.0, 0
    for rcg, posxyz, ecg in test_loader:
        rcg, posxyz, ecg = rcg.to(device), posxyz.to(device), ecg.to(device)
        companded_pred = model.generate(rcg, posxyz)
        pred = mu_law_decode(companded_pred)
        total_corr += _pearson_corr(pred, ecg)
        n_batches += 1

    return {"corr": total_corr / max(n_batches, 1), "loss": float("nan")}
