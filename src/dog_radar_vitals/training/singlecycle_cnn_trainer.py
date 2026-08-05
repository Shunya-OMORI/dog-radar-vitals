"""単一拍SST->ECG回帰の凡庸なCNNベースライン学習ループ。family=`mmecg_singlecycle_cnn`。

`radarode_sceg_trainer.py`とほぼ同一の学習ループ(同じDataset・同じSampler・同じ評価粒度)だが、
モデルを`SingleCycleCNN`に差し替えている。SCEGと全く同じタスク・データパイプラインの上で
モデルだけを凡庸なCNNに変えることで、「アーキテクチャの工夫がどれだけ寄与しているか」を
切り分けて評価できる(`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
「タスク設計・前処理・学習設計の改善」節参照)。

## Shift-invariant training(2026-07-30)

学習可能なτ補正(`TemporalShiftHead`)は入力のみからは有用なシフト量を学習できず、
ほぼ定数に収束することを確認した。一方、正解を知っているoracle shift(全探索)を適用すると
257(シフト無し)の相関は0.203→0.586まで跳ね上がることも確認しており、「正確な個別補正には
大きな伸びしろがある」こと自体は裏付けられている。`train.oracle_shift_train: true`で、
**学習時のみ**(推論・評価時は使わない)各バッチの予測をoracle shiftで正解に最も近づけてから
損失を計算する(`temporal_alignment.oracle_best_shift`)。MSE損失がタイミング不確実性を
理由に予測を鈍らせるインセンティブを、学習時のアライメントで取り除けるかを検証する
(評価は常に生の予測に対して行うため、257等の既存configとそのまま比較できる)。`oracle_max_shift`
を小さくする(例:±30→±5)ことで、shift-invariant訓練が生出力の位置合わせ動機を完全に
失わないよう制約できる(259の失敗の直接的な対策候補)。

## τの教師あり補助損失(tau supervision, 2026-07-30)

259で無制約のshift-invariant trainingが失敗した(train_lossは下がるがcorrが大幅悪化)ことを
受け、`TemporalShiftHead`が予測するτ自体を、固定した参照モデル(通常は257=シフト無し
コントロール)の予測に対してoracle探索で得た擬似ラベルτへ直接回帰させる補助損失を追加した。
`train.tau_supervision`で有効化する:
```yaml
train:
  tau_supervision:
    enabled: true
    source_run_dir: runs/20260730-113005_ecg_singlecycle_cnn  # 257(擬似ラベル生成用の固定モデル)
    max_shift: 30
    weight: 1.0
```
学習開始前に、`source_run_dir`のモデルで train set 全体のoracleシフトを一度だけ計算し
(`_compute_oracle_tau_labels`)、データセットの`__getitem__`インデックスに紐付けて保存する。
学習中は`TrialInterleavedSampler`でシャッフルされてもインデックスさえ分かればラベルを
引けるよう、`_IndexedDataset`でラベル探索用のインデックスを一緒に返す。損失は
`reconstruction_loss(shifted_out, ecg) + weight * MSE(predicted_tau, pseudo_label_tau)`。
評価は常にモデル自身が予測したτ(擬似ラベルではない)を使った生の推論経路で行う。
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
from dog_radar_vitals.models.deep.singlecycle_cnn import SingleCycleCNN
from dog_radar_vitals.models.deep.temporal_alignment import oracle_best_shift, shift_batch_per_example
from dog_radar_vitals.seeding import make_generator, set_all_seeds
from dog_radar_vitals.training.ecg_trainer import _pearson_corr
from dog_radar_vitals.training.schedulers import build_scheduler

SST_CACHE_SIZE = 4


class _IndexedDataset(Dataset):
    """base[idx] = (sst, ecg)に、`__getitem__`インデックス自体を第3要素として付加するラッパー。

    τ擬似ラベルは`base`の`__getitem__`インデックス順に事前計算しているため、
    `TrialInterleavedSampler`でシャッフルされた後もこのインデックスでラベルを引ける。
    """

    def __init__(self, base: MMECGSingleCycleDataset) -> None:
        self.base = base
        self.trial_of_index = base.trial_of_index  # サンプラーが参照する

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sst, ecg = self.base[idx]
        return sst, ecg, idx


def _compute_oracle_tau_labels(
    source_run_dir: Path, dataset: MMECGSingleCycleDataset, device: torch.device, max_shift: int
) -> torch.Tensor:
    """`source_run_dir`の固定モデル(通常は257=シフト無しコントロール)で`dataset`全体を
    順番に(シャッフルせず)推論し、各例についてoracle探索で得た最良シフトをτ_frac
    (=−shift/out_len、`shift_signal`の符号規約に合わせる)に変換して返す。
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


def _build_dataset(data_cfg: dict, raw_root: Path, split: str) -> MMECGSingleCycleDataset:
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])
    return MMECGSingleCycleDataset(
        raw_root,
        trial_ids,
        cache_size=SST_CACHE_SIZE,
        norm_scope=data_cfg.get("norm_scope", "trial"),
        # 既定False: 247等は`apply_bandpass`概念導入前に学習されており、config.yamlに
        # このキーが無い。デフォルトをTrueにすると、それらのrunをevaluate.pyで再評価した際に
        # 学習時(バンドパスなし)と異なる設定でtest setが作られてしまう(train/eval不整合)。
        # 250以降はconfigに明示的にキーを書くため、この既定値の影響を受けない。
        apply_bandpass=data_cfg.get("apply_bandpass", False),
        use_context_window=data_cfg.get("use_context_window", False),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer=None,
    oracle_shift_train: bool = False,
    oracle_max_shift: int = 30,
    tau_labels: torch.Tensor | None = None,
    tau_weight: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_corr, n_batches = 0.0, 0.0, 0
    total_tau_loss = 0.0
    loss_fn = nn.MSELoss()

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            if tau_labels is not None:
                sst, ecg, idx = batch
            else:
                sst, ecg = batch
            sst, ecg = sst.to(device), ecg.to(device)

            if is_train and tau_labels is not None:
                pred, tau = model(sst, return_tau=True)
                tau_target = tau_labels[idx].to(device)
                recon_loss = loss_fn(pred, ecg)
                tau_loss = loss_fn(tau, tau_target)
                loss = recon_loss + tau_weight * tau_loss
                total_tau_loss += tau_loss.item()
            else:
                pred = model(sst)

                if is_train and oracle_shift_train:
                    # oracle shiftはevalモードでも使えるが、学習時のみ適用(推論・評価は常に生の
                    # 予測に対して行い、既存configとそのまま比較できるようにする)。
                    with torch.no_grad():
                        best_shift, _ = oracle_best_shift(pred, ecg, oracle_max_shift)
                    pred_for_loss = shift_batch_per_example(pred, best_shift)
                else:
                    pred_for_loss = pred

                loss = loss_fn(pred_for_loss, ecg)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_corr += _pearson_corr(pred.detach(), ecg)  # 生の予測で評価(常に一貫比較)
            n_batches += 1

    metrics = {"loss": total_loss / n_batches, "corr": total_corr / n_batches}
    if tau_labels is not None and is_train:
        metrics["tau_loss"] = total_tau_loss / n_batches
    return metrics


def train_singlecycle_cnn(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    train_ds = _build_dataset(data_cfg, raw_root, "train")
    val_ds = _build_dataset(data_cfg, raw_root, "val")
    print(f"train cycles: {len(train_ds)}, val cycles: {len(val_ds)}")

    train_cfg = config["train"]

    tau_sup_cfg = train_cfg.get("tau_supervision", {})
    tau_sup_enabled = tau_sup_cfg.get("enabled", False)
    tau_labels = None
    tau_weight = tau_sup_cfg.get("weight", 1.0)
    if tau_sup_enabled:
        source_run_dir = repo_root / tau_sup_cfg["source_run_dir"]
        tau_max_shift = tau_sup_cfg.get("max_shift", 30)
        print(f"computing oracle tau pseudo-labels from {source_run_dir} (max_shift={tau_max_shift})...")
        tau_labels = _compute_oracle_tau_labels(source_run_dir, train_ds, device, tau_max_shift)
        print(
            f"tau labels: mean={tau_labels.mean():.4f} std={tau_labels.std():.4f} "
            f"mean_abs={tau_labels.abs().mean():.4f}"
        )
        train_ds = _IndexedDataset(train_ds)

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
    model = SingleCycleCNN(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = build_scheduler(optimizer, train_cfg, total_epochs=train_cfg["epochs"])

    oracle_shift_train = train_cfg.get("oracle_shift_train", False)
    oracle_max_shift = train_cfg.get("oracle_max_shift", 30)

    history = []
    best_val_corr = -float("inf")
    for epoch in range(train_cfg["epochs"]):
        if scheduler is not None:
            scheduler.step()
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            oracle_shift_train=oracle_shift_train,
            oracle_max_shift=oracle_max_shift,
            tau_labels=tau_labels,
            tau_weight=tau_weight,
        )
        val_metrics = _run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        tau_loss_str = f" tau_loss={train_metrics['tau_loss']:.4f}" if "tau_loss" in train_metrics else ""
        print(
            f"[epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_corr={train_metrics['corr']:.3f}{tau_loss_str} "
            f"val_loss={val_metrics['loss']:.4f} val_corr={val_metrics['corr']:.3f}"
        )

        if val_metrics["corr"] > best_val_corr:
            best_val_corr = val_metrics["corr"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")

    (run_dir / "metrics.json").write_text(json.dumps({"history": history, "best_val_corr": best_val_corr}, indent=2))


def evaluate_singlecycle_cnn(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    test_ds = _build_dataset(data_cfg, raw_root, "test")
    test_loader = DataLoader(test_ds, batch_size=config["train"]["batch_size"], shuffle=False)

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = SingleCycleCNN(**model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))

    metrics = _run_epoch(model, test_loader, device)
    return {"corr": metrics["corr"], "loss": metrics["loss"]}
