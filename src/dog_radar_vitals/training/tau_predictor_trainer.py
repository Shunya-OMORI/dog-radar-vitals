"""τ専用回帰モデル(`TauOnlyCNN`)の学習ループ。family=`mmecg_tau_predictor`。

261/262(波形回帰モデルにτ教師あり補助損失を追加)はいずれも、予測τがoracle疑似ラベルτと
ほぼ無相関のまま(残差学習信号がほぼゼロ)だった。「波形回帰に特化したアーキテクチャの
中間特徴からτを予測する」という設計そのものがτ推定に向いていない可能性と、
「タスクとして原理的にτを個別に予測することが困難(疑似ラベル自体がノイズ)」という
可能性を切り分けるため、波形は一切予測せずτのみを回帰する専用モデルを、τ専用の
教師あり損失のみで学習する(`singlecycle_cnn_trainer.py`のτ補助損失のような
「主タスクの片手間」ではない、純粋なτ回帰タスクとしての学習)。

擬似ラベルτは`singlecycle_cnn_trainer._compute_oracle_tau_labels`と同じ手法
(固定した参照モデル=257の予測に対するoracle探索)で、train/val/test全splitについて
学習開始前に一括計算する。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_singlecycle_dataset import (
    MMECGSingleCycleDataset,
    TrialInterleavedSampler,
)
from dog_radar_vitals.data.tau_pseudo_labels import compute_rpeak_position_tau_labels
from dog_radar_vitals.models.deep.singlecycle_cnn import SingleCycleCNN
from dog_radar_vitals.models.deep.tau_predictor import TauOnlyCNN
from dog_radar_vitals.models.deep.temporal_alignment import oracle_best_shift
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.schedulers import build_scheduler

SST_CACHE_SIZE = 4


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGSingleCycleDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSingleCycleDataset(
        raw_root,
        trial_ids,
        cache_size=SST_CACHE_SIZE,
        norm_scope=data_cfg.get("norm_scope", "trial"),
        apply_bandpass=data_cfg.get("apply_bandpass", False),
        use_context_window=data_cfg.get("use_context_window", False),
    )


class _TauLabeledDataset(Dataset):
    """(sst, ecg)を返す`base`を、(sst, tau_label)を返すデータセットにラップする。"""

    def __init__(self, base: MMECGSingleCycleDataset, tau_labels: torch.Tensor) -> None:
        assert len(base) == len(tau_labels)
        self.base = base
        self.tau_labels = tau_labels
        self.trial_of_index = base.trial_of_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sst, _ecg = self.base[idx]
        return sst, self.tau_labels[idx]


def _compute_oracle_tau_labels(
    source_run_dir: Path, dataset: MMECGSingleCycleDataset, device: torch.device, max_shift: int
) -> torch.Tensor:
    """固定した波形回帰モデル(通常は257)の予測に対するoracle探索でτ疑似ラベルを作る。
    `singlecycle_cnn_trainer._compute_oracle_tau_labels`と同一ロジック(独立した波形モデルに
    依存させないため、ここでも同じ計算を行う。将来的に共通化してもよい)。
    """
    with open(source_run_dir / "config.yaml") as f:
        source_cfg = yaml.safe_load(f)
    model_kwargs = {k: v for k, v in source_cfg["model"].items() if k not in ("family", "name")}
    model_kwargs["use_temporal_shift"] = False
    source_model = SingleCycleCNN(**model_kwargs).to(device)
    source_model.load_state_dict(torch.load(source_run_dir / "best_model.pt", map_location=device))
    source_model.eval()

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    tau_chunks = []
    with torch.no_grad():
        for sst, ecg in loader:
            sst, ecg = sst.to(device), ecg.to(device)
            pred = source_model(sst)
            best_shift, _ = oracle_best_shift(pred, ecg, max_shift)
            out_len = ecg.shape[-1]
            tau_chunks.append((-best_shift.float() / out_len).cpu())
    return torch.cat(tau_chunks, dim=0)


def _compute_tau_labels(
    tau_cfg: dict, dataset: MMECGSingleCycleDataset, device: torch.device, repo_root: Path
) -> torch.Tensor:
    """`tau_cfg["method"]`に応じて疑似ラベルを計算する。

    - `"oracle_shift"`(既定、後方互換): 257等の固定参照モデルの予測に対するoracle探索。
    - `"rpeak_position"`: `tau_pseudo_labels.py`参照。モデルに依存しない、拍境界内でのR波位置。
    """
    method = tau_cfg.get("method", "oracle_shift")
    if method == "rpeak_position":
        return compute_rpeak_position_tau_labels(dataset)
    elif method == "oracle_shift":
        source_run_dir = repo_root / tau_cfg["source_run_dir"]
        max_shift = tau_cfg.get("max_shift", 30)
        return _compute_oracle_tau_labels(source_run_dir, dataset, device, max_shift)
    else:
        raise ValueError(f"unknown tau label method: {method}")


def _pearson_corr_scalar(pred: torch.Tensor, target: torch.Tensor) -> float:
    pc = pred - pred.mean()
    tc = target - target.mean()
    denom = pc.norm() * tc.norm() + 1e-8
    return ((pc * tc).sum() / denom).item()


def _run_epoch(
    model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.MSELoss()

    total_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(is_train):
        for sst, tau_label in loader:
            sst, tau_label = sst.to(device), tau_label.to(device)
            pred_tau = model(sst)
            loss = loss_fn(pred_tau, tau_label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(pred_tau.detach().cpu())
            all_labels.append(tau_label.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    mae = (all_preds - all_labels).abs().mean().item()
    corr = _pearson_corr_scalar(all_preds, all_labels)
    return {
        "loss": total_loss / n_batches,
        "mae": mae,
        "corr": corr,
        "pred_std": all_preds.std().item(),
        "label_std": all_labels.std().item(),
    }


def train_tau_predictor(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    tau_cfg = config["train"]["tau_labels"]
    method = tau_cfg.get("method", "oracle_shift")

    train_ds_raw = _build_dataset(data_cfg, raw_root, "train")
    val_ds_raw = _build_dataset(data_cfg, raw_root, "val")
    print(f"train cycles: {len(train_ds_raw)}, val cycles: {len(val_ds_raw)}")

    print(f"computing tau pseudo-labels (method={method})...")
    train_tau = _compute_tau_labels(tau_cfg, train_ds_raw, device, repo_root)
    val_tau = _compute_tau_labels(tau_cfg, val_ds_raw, device, repo_root)
    print(
        f"train tau labels: mean={train_tau.mean():.4f} std={train_tau.std():.4f} "
        f"mean_abs={train_tau.abs().mean():.4f}"
    )

    train_ds = _TauLabeledDataset(train_ds_raw, train_tau)
    val_ds = _TauLabeledDataset(val_ds_raw, val_tau)

    train_cfg = config["train"]
    train_sampler = TrialInterleavedSampler(
        train_ds.trial_of_index, pool_size=SST_CACHE_SIZE, shuffle=True, seed=config["seed"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        generator=make_generator(config["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = TauOnlyCNN(**model_kwargs).to(device)
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
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.5f} train_corr={train_metrics['corr']:.3f} "
            f"train_pred_std={train_metrics['pred_std']:.4f} "
            f"val_loss={val_metrics['loss']:.5f} val_corr={val_metrics['corr']:.3f} "
            f"val_pred_std={val_metrics['pred_std']:.4f}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_tau_predictor(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    tau_cfg = config["train"]["tau_labels"]

    test_ds_raw = _build_dataset(data_cfg, raw_root, "test")
    test_tau = _compute_tau_labels(tau_cfg, test_ds_raw, device, repo_root)
    test_ds = _TauLabeledDataset(test_ds_raw, test_tau)
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = TauOnlyCNN(**model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    # primary_key探索の都合で"corr"を主要指標として使う(evaluate.pyのprimary_key選択と合わせる)。
    return {"corr": metrics["corr"], "loss": metrics["loss"], "mae": metrics["mae"], "pred_std": metrics["pred_std"]}
