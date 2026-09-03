"""RCG+posXYZ(空間座標) -> ECG波形/R波heatmap の学習ループ。family=`mmecg_spatial_seq2seq`。

`ecg_spatial_fusion.py`/`rpeak_spatial_fusion.py`は`forward(rcg, posxyz)`と2引数を取るため、
1引数`forward(x)`を前提とする`models/deep/ecg_registry.py`には登録せず、本ファイル内の
小さなレジストリで扱う（既存のfamilyとは呼び出し規約が異なるため、AGENTS.mdの規約に従い
別ファイル・別registryとする）。損失関数は`model.name`に応じてMSE(波形)かBCE(heatmap)を
自動選択する。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset
from dog_radar_vitals.models.deep.ecg_spatial_fusion import ECGSpatialFusion
from dog_radar_vitals.models.deep.ecg_spatial_gnn import ECGSpatialGNN
from dog_radar_vitals.models.deep.rpeak_spatial_fusion import RPeakSpatialFusion
from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.losses import (
    adaptive_wing_loss,
    box_bce_dice_loss,
    heatmap_bce_localexp_loss,
    heatmap_focal_loss,
    heatmap_weighted_bce_loss,
)
from dog_radar_vitals.training.schedulers import build_scheduler

_SPATIAL_MODEL_REGISTRY = {
    "ecg_spatial_fusion": ECGSpatialFusion,
    "rpeak_spatial_fusion": RPeakSpatialFusion,
    "ecg_spatial_gnn": ECGSpatialGNN,
    "rpeak_spatial_gnn": RPeakSpatialGNN,
}
_IS_HEATMAP = {
    "ecg_spatial_fusion": False,
    "rpeak_spatial_fusion": True,
    "ecg_spatial_gnn": False,
    "rpeak_spatial_gnn": True,
}


def build_spatial_model(model_cfg: dict) -> nn.Module:
    """`mmecg_rpeak_evaluation.py`からも使う公開ビルダ（2引数forward(rcg, posxyz)を取るモデル用）。
    `mmecg_spatial_heatmap_gan`のconfigには判別器用の`discriminator`/`adv_weight`キーが同居する
    ため、生成器のコンストラクタには渡さないよう常に除外する。
    """
    kwargs = {k: v for k, v in model_cfg.items() if k not in ("family", "name", "discriminator", "adv_weight")}
    return _SPATIAL_MODEL_REGISTRY[model_cfg["name"]](**kwargs)


def is_heatmap_model(name: str) -> bool:
    return _IS_HEATMAP[name]


def _build_loss_fn(model_name: str, train_cfg: dict):
    """`train.loss`（既定"bce"）でheatmapモデルの損失を切り替える。波形モデルは常にMSE。

    2026-08-06追記: 後処理（極大点の選択）は生成済みheatmapの中から選ぶことしかできず、
    真のR波位置でheatmap値が十分上がっていない（見逃し）場合は取り返せない。BCEは
    適合率・再現率を対称に扱うため、"focal"（`losses.heatmap_focal_loss`、CornerNet/
    CenterNet由来）で見逃し防止側に非対称な重み付けを試せるようにする。
    """
    if not _IS_HEATMAP[model_name]:
        return nn.MSELoss()
    loss_name = train_cfg.get("loss", "bce")
    if loss_name == "focal":
        return heatmap_focal_loss
    if loss_name == "weighted_bce":
        # 見逃しを重く罰する (2026-08-25)。HRV を目的にすると、拍の見逃しは
        # その区間の RR 間隔を倍にして RMSSD を壊すため、誤検出より害が大きい。
        pos_weight = float(train_cfg.get("pos_weight", 10.0))
        return lambda pred, y: heatmap_weighted_bce_loss(pred, y, pos_weight=pos_weight)
    if loss_name == "bce_localexp":
        loc_weight = float(train_cfg.get("loc_weight", 0.1))
        half_win = int(train_cfg.get("loc_half_win", 17))
        return lambda pred, y: heatmap_bce_localexp_loss(pred, y, loc_weight=loc_weight, half_win=half_win)
    if loss_name == "adaptive_wing":
        # 2026-08-28: Wang ら(ICCV 2019)。BCE系ではなくL1/Wing系の損失族との比較用。
        return adaptive_wing_loss
    if loss_name == "box_bce_dice":
        # 2026-08-28: build_peak_box(矩形マスク教師)専用。QRS検出のU-Net系文献の定番構成。
        dice_weight = float(train_cfg.get("dice_weight", 1.0))
        return lambda pred, y: box_bce_dice_loss(pred, y, dice_weight=dice_weight)
    return nn.BCELoss()


_build_model = build_spatial_model


def _build_dataset(data_cfg: dict, raw_root: Path, split: str, heatmap: bool) -> MMECGSpatialWindowDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSpatialWindowDataset(
        raw_root,
        trial_ids,
        data_cfg["window_sec"],
        data_cfg["stride_sec"],
        heatmap=heatmap,
        normalization=data_cfg.get("normalization", "zscore"),
        rpeak_detector=data_cfg.get("rpeak_detector", "legacy"),
        apply_bandpass=data_cfg.get("apply_bandpass", False),
        preprocess=data_cfg.get("preprocess", "none"),
        heatmap_sigma_ms=data_cfg.get("heatmap_sigma_ms", 10.0),
        target_mode=data_cfg.get("target_mode", "all_peaks"),
        target_shape=data_cfg.get("target_shape", "gaussian"),
        box_half_width_ms=data_cfg.get("box_half_width_ms", 150.0),
    )


def _run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn, optimizer=None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, n_batches = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for rcg, posxyz, y in loader:
            rcg, posxyz, y = rcg.to(device), posxyz.to(device), y.to(device)
            pred = model(rcg, posxyz)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), y)
            n_batches += 1

    return {"loss": total_loss / n_batches, "corr": total_corr / n_batches}


def train_spatial_fusion(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    is_heatmap = _IS_HEATMAP[config["model"]["name"]]

    train_ds = _build_dataset(data_cfg, raw_root, "train", is_heatmap)
    val_ds = _build_dataset(data_cfg, raw_root, "val", is_heatmap)

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
    loss_fn = _build_loss_fn(config["model"]["name"], train_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_metrics = _run_epoch(model, train_loader, device, loss_fn, optimizer)
        val_metrics = _run_epoch(model, val_loader, device, loss_fn)
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics})
        print(
            f"[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.2e} train_loss={train_metrics['loss']:.4f} "
            f"train_corr={train_metrics['corr']:.3f} val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}"
        )

        # 2026-08-07: チェックポイント選択基準(val_corr、heatmap全体のPearson相関)が
        # 実際の下流タスク(ピーク検出のF1・RR/RMSSD精度)と乖離する場合がある実例を確認した
        # (R+T統合教師モデルで、corrが同水準でも下流精度が大きく異なった)。best_model.pt
        # だけでは事後的に「本当に良かったepoch」を再評価できないため、一定間隔でも
        # チェックポイントを残し、post-hocに下流タスクで選び直せるようにする。
        if (epoch + 1) % 5 == 0 or epoch == train_cfg["epochs"] - 1:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_spatial_fusion(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    is_heatmap = _IS_HEATMAP[config["model"]["name"]]
    test_ds = _build_dataset(data_cfg, raw_root, "test", is_heatmap)
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model = _build_model(config["model"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    loss_fn = _build_loss_fn(config["model"]["name"], config["train"])

    metrics = _run_epoch(model, test_loader, device, loss_fn)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
