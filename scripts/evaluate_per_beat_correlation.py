"""Chen et al. (2022, RCG2ECG)の相関評価方法を再現した波形相関の再評価。

原論文は「reconstructed waveform segmented by the ground truth heart beat cycles」ごとに
Pearson相関を計算し、その中央値を報告している（本文4.2.3節）。一方このリポジトリの
`_pearson_corr`（`training/ecg_trainer.py`）は窓全体(4秒、複数拍分)に対して1回だけ
相関を計算しており、評価粒度が異なる。0.224 vs 0.90という差のどこまでが「モデルの実力差」で
どこまでが「評価粒度の違い」かを切り分けるため、同一モデルを2つの粒度で再評価する。

使い方:
    python scripts/evaluate_per_beat_correlation.py runs/<run_dir> [runs/<run_dir> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.mmecg import trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.mmecg_dataset import MMECGWindowDataset  # noqa: E402
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset  # noqa: E402
from dog_radar_vitals.data.mulaw import mu_law_decode  # noqa: E402
from dog_radar_vitals.data.rpeaks import detect_r_peaks  # noqa: E402
from dog_radar_vitals.models.deep.chen2022_reconstructor import Chen2022Reconstructor  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402


def per_beat_correlations(pred: np.ndarray, true: np.ndarray, fs: int) -> list[float]:
    """1窓分の(pred, true)波形をtrueのR波位置で拍単位に区切り、拍ごとの相関係数を返す。

    R波検出はtrue波形（正規化済みだがdetect_r_peaks内部で再z-score化するのでスケール非依存）
    で行い、隣接R波の中間点を拍の境界にする（1拍=前後の谷を跨がずR波を中心に取る近似）。
    3サンプル未満の拍・分散ゼロの拍は相関が定義できないためスキップする。
    """
    peaks = detect_r_peaks(true, fs)
    if len(peaks) < 2:
        return []
    boundaries = [0] + [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)] + [len(true)]
    corrs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 3:
            continue
        t_seg, p_seg = true[start:end], pred[start:end]
        if np.std(t_seg) < 1e-6 or np.std(p_seg) < 1e-6:
            continue
        corrs.append(float(np.corrcoef(t_seg, p_seg)[0, 1]))
    return corrs


def _predict_batch_chen2022(model: Chen2022Reconstructor, rcg, posxyz, device: torch.device) -> np.ndarray:
    """自己回帰生成でECGを再構成し、companding空間から元の振幅へ逆変換して返す(numpy, batch分)。"""
    companded = model.generate(rcg.to(device), posxyz.to(device))
    return mu_law_decode(companded).cpu().numpy()


def evaluate_run(run_dir: Path) -> None:
    config = load_config(run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]
    raw_root = REPO_ROOT / data_cfg["raw_root"]
    family = config["model"]["family"]
    is_chen2022 = family == "mmecg_chen2022"

    fs = 200  # MMECG固定サンプリングレート（data/mmecg.py FS_HZ）
    window_corrs: list[float] = []
    beat_corrs: list[float] = []

    if is_chen2022:
        trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"]["test"])
        test_ds = MMECGSpatialWindowDataset(
            raw_root,
            trial_ids,
            data_cfg["window_sec"],
            data_cfg["stride_sec"],
            heatmap=False,
            normalization=data_cfg.get("normalization", "minmax"),
        )
        model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
        model = Chen2022Reconstructor(**model_kwargs).to(device)
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
        model.eval()

        loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
        with torch.no_grad():
            for rcg, posxyz, ecg in loader:
                pred_batch = _predict_batch_chen2022(model, rcg, posxyz, device)
                true_batch = ecg.numpy()
                for pred, true in zip(pred_batch, true_batch):
                    if np.std(pred) > 1e-6 and np.std(true) > 1e-6:
                        window_corrs.append(float(np.corrcoef(pred, true)[0, 1]))
                    beat_corrs.extend(per_beat_correlations(pred, true, fs))
    else:
        trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"]["test"])
        test_ds = MMECGWindowDataset(
            raw_root,
            trial_ids,
            data_cfg["window_sec"],
            data_cfg["stride_sec"],
            complex_input=data_cfg.get("complex_input", False),
            normalization=data_cfg.get("normalization", "zscore"),
        )

        model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
        model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
        model.eval()

        with torch.no_grad():
            for i in range(len(test_ds)):
                x, y = test_ds[i]
                pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                true = y.numpy()

                if np.std(pred) > 1e-6 and np.std(true) > 1e-6:
                    window_corrs.append(float(np.corrcoef(pred, true)[0, 1]))
                beat_corrs.extend(per_beat_correlations(pred, true, fs))

    name = f"{config['model']['name']} ({data_cfg.get('normalization', 'zscore')})"
    print(f"\n=== {name}  [{run_dir}] ===")
    print(f"  window-level  : mean={np.mean(window_corrs):.3f}  median={np.median(window_corrs):.3f}  n={len(window_corrs)}")
    print(f"  per-beat(Chen式): mean={np.mean(beat_corrs):.3f}  median={np.median(beat_corrs):.3f}  n={len(beat_corrs)}")
    print(f"                  90th percentile={np.percentile(beat_corrs, 90):.3f}  10th percentile={np.percentile(beat_corrs, 10):.3f}")


if __name__ == "__main__":
    for run_arg in sys.argv[1:]:
        evaluate_run(Path(run_arg))
