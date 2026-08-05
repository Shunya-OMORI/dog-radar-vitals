"""radarODEのSCEG(単一拍ECG再構成)の学習ループ。family=`mmecg_radarode_sceg`。

評価指標はRMSE・Pearson相関(PCC)を1拍ごとに計算して平均する。radarODE論文Table VIの
「MMECG再現87.9%・radarODE自身92.6%」という数値と直接比較できるよう、この単一拍単位の
評価にとどめる（`_pearson_corr`のような窓内評価ではなく、原著と同じ粒度で比較するため）。

## 相関損失オプション(2026-07-30)

248(MSE損失)の予測を診断したところ、±20サンプル(200サンプル中)のシフト探索を許すと
相関が0.36→0.54まで跳ね上がることを確認した(シフト方向は左右対称、系統的な生理学的遅延
ではなく拍境界推定のノイズによると推測、詳細は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。これは、MSE損失が
QRSのような鋭いピークに対して「タイミング不確実性をヘッジするため予測を鈍らせる」構造的な
問題(既報告の「QRS振幅の鈍り」)を引き起こしている可能性を示唆する。`train.loss_type: "corr"`
で、MSEの代わりに微分可能なPearson相関損失(`1 - corr`)を使えるようにした
(既定は`"mse"`で従来通り)。評価指標(相関)と学習目的を直接一致させることで、
この鈍化インセンティブを取り除く狙い。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dog_radar_vitals.data.mmecg import trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_singlecycle_dataset import (
    MMECGSingleCycleDataset,
    TrialInterleavedSampler,
)
from dog_radar_vitals.data.tau_pseudo_labels import compute_rpeak_heatmap_labels
from dog_radar_vitals.models.deep.radarode_sceg import RadarODESCEG
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.schedulers import build_scheduler


class _AnchorLabeledDataset:
    """(sst, ecg)を返す`base`を、(sst, ecg, anchor_heatmap)を返すデータセットにラップする。

    `use_anchor_head=True`時、radarODE公式実装の"Anchor"補助タスク(R波位置ヒートマップ回帰、
    `tau_pseudo_labels.compute_rpeak_heatmap_labels`参照)を追加で学習するために使う。
    """

    def __init__(self, base: MMECGSingleCycleDataset, anchor_labels: torch.Tensor) -> None:
        assert len(base) == len(anchor_labels)
        self.base = base
        self.anchor_labels = anchor_labels
        self.trial_of_index = base.trial_of_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sst, ecg = self.base[idx]
        return sst, ecg, self.anchor_labels[idx]


def _pearson_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """`1 - (バッチ内各サンプルのPearson相関の平均)`。微分可能、`_pearson_corr`と対応。"""
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    true_c = true - true.mean(dim=1, keepdim=True)
    numerator = (pred_c * true_c).sum(dim=1)
    denominator = pred_c.norm(dim=1) * true_c.norm(dim=1) + 1e-8
    corr = numerator / denominator
    return (1 - corr).mean()


def _build_loss_fn(loss_type: str):
    if loss_type == "mse":
        mse = nn.MSELoss()
        return lambda pred, true: mse(pred, true)
    if loss_type == "corr":
        return _pearson_loss
    raise ValueError(f"unknown loss_type: {loss_type}")

SST_CACHE_SIZE = 4  # MMECGSingleCycleDatasetのLRUキャッシュサイズ。TrialInterleavedSamplerの
# pool_sizeと揃えること(揃えないとキャッシュがpool_size分のトライアルを保持しきれず、
# ワーカーごとに余分なnpz再ロードが発生する)。


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGSingleCycleDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSingleCycleDataset(
        raw_root,
        trial_ids,
        cache_size=SST_CACHE_SIZE,
        norm_scope=data_cfg.get("norm_scope", "trial"),
        # 既定False: 245-249は`apply_bandpass`概念導入前に学習されており、config.yamlに
        # このキーが無い。デフォルトをTrueにすると、それらのrunをevaluate.pyで再評価した際に
        # 学習時(バンドパスなし)と異なる設定でtest setが作られてしまう(train/eval不整合)。
        # 253以降はconfigに明示的に`apply_bandpass: true`を書くため、この既定値の影響を受けない。
        apply_bandpass=data_cfg.get("apply_bandpass", False),
        use_context_window=data_cfg.get("use_context_window", False),
        t_fixed_sst=data_cfg.get("t_fixed_sst"),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    loss_type: str = "mse",
    use_anchor: bool = False,
    anchor_weight: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, total_anchor_loss, n_batches = 0.0, 0.0, 0.0, 0
    loss_fn = _build_loss_fn(loss_type)
    anchor_loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if use_anchor:
                sst, ecg, anchor_gt = batch
                anchor_gt = anchor_gt.to(device)
            else:
                sst, ecg = batch

            sst, ecg = sst.to(device), ecg.to(device)

            if use_anchor:
                pred, _tau, anchor_pred = model(sst, return_anchor=True)
                shape_loss = loss_fn(pred, ecg)
                anchor_loss = anchor_loss_fn(anchor_pred, anchor_gt)
                loss = shape_loss + anchor_weight * anchor_loss
            else:
                pred = model(sst)
                loss = loss_fn(pred, ecg)
                anchor_loss = None

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), ecg)
            if anchor_loss is not None:
                total_anchor_loss += anchor_loss.item()
            n_batches += 1

    metrics = {"loss": total_loss / n_batches, "corr": total_corr / n_batches}
    if use_anchor:
        metrics["anchor_loss"] = total_anchor_loss / n_batches
    return metrics


def train_radarode_sceg(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    train_ds = _build_dataset(data_cfg, raw_root, "train")
    val_ds = _build_dataset(data_cfg, raw_root, "val")
    print(f"train cycles: {len(train_ds)}, val cycles: {len(val_ds)}")

    train_cfg = config["train"]
    use_anchor = config["model"].get("use_anchor_head", False)
    anchor_weight = train_cfg.get("anchor_weight", 1.0)
    trial_of_index = train_ds.trial_of_index
    if use_anchor:
        print("computing anchor(R波位置ヒートマップ)疑似ラベル...")
        train_anchor = compute_rpeak_heatmap_labels(train_ds)
        val_anchor = compute_rpeak_heatmap_labels(val_ds)
        train_ds = _AnchorLabeledDataset(train_ds, train_anchor)
        val_ds = _AnchorLabeledDataset(val_ds, val_anchor)

    train_sampler = TrialInterleavedSampler(
        trial_of_index, pool_size=SST_CACHE_SIZE, shuffle=True, seed=config["seed"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        generator=make_generator(config["seed"]),
    )
    # valはインデックスが既にtrial単位で連続しているため(__init__でtrial順に追加)、
    # sequential(shuffle=False)のままでもキャッシュ局所性は保たれる。
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = RadarODESCEG(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])
    loss_type = train_cfg.get("loss_type", "mse")

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_metrics = _run_epoch(
            model, train_loader, device, optimizer, loss_type=loss_type, use_anchor=use_anchor, anchor_weight=anchor_weight
        )
        val_metrics = _run_epoch(
            model, val_loader, device, loss_type=loss_type, use_anchor=use_anchor, anchor_weight=anchor_weight
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        anchor_str = (
            f" train_anchor_loss={train_metrics['anchor_loss']:.4f} val_anchor_loss={val_metrics['anchor_loss']:.4f}"
            if use_anchor
            else ""
        )
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_corr={train_metrics['corr']:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}{anchor_str}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_radarode_sceg(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    use_anchor = config["model"].get("use_anchor_head", False)
    if use_anchor:
        test_anchor = compute_rpeak_heatmap_labels(test_ds)
        test_ds = _AnchorLabeledDataset(test_ds, test_anchor)
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = RadarODESCEG(**model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(
        model,
        test_loader,
        device,
        loss_type=config["train"].get("loss_type", "mse"),
        use_anchor=use_anchor,
        anchor_weight=config["train"].get("anchor_weight", 1.0),
    )
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
