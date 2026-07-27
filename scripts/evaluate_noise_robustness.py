"""学習済みの波形回帰モデル(ecg_ncp / ecg_cnn1d / ecg_unet1d等)に、テスト時にRCG入力へ
段階的にガウスノイズを注入し、相関係数の劣化曲線を比較する。

文献調査（`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「NCPを別軸で評価する」
参照）でLiquid Neural Networks(NCP/LTCの系統)が「同条件のノイズ下でRNN/CNNより性能劣化が
緩やか」と報告されていることを、本データセットで直接検証する。再学習は不要（既存の
best_model.ptをそのまま評価に使う）。

RCGは`data/mmecg_windowing.zscore_channels`でチャネルごとにz-score正規化済み（std≈1）なため、
`noise_std`は「信号の標準偏差に対する相対的なノイズ強度」として解釈できる
（0.5なら信号の半分程度のノイズ、1.0なら信号と同程度、2.0ならノイズが信号の2倍で
ほぼ埋もれる、という目安）。

使い方:
    python scripts/evaluate_noise_robustness.py --runs runs/xxx_ecg_ecg_ncp runs/yyy_ecg_ecg_cnn1d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from torch.utils.data import DataLoader

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.mmecg import trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.mmecg_dataset import MMECGWindowDataset  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402
from dog_radar_vitals.training.ecg_trainer import _pearson_corr  # noqa: E402

NOISE_STDS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
N_SEEDS_PER_LEVEL = 3  # ノイズ自体の乱数シードを変えて平均する（1回だけだと運の影響が大きいため）
BATCH_SIZE = 64  # NCPのCfCは時間方向に逐次実行されるため、バッチ化で1サンプルあたりのコストを分散する


def _load_model(run_dir: Path, config: dict) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


def _evaluate_at_noise_level(model: torch.nn.Module, loader: DataLoader, noise_std: float, seed: int) -> float:
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(seed)

    total_corr, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            if noise_std > 0:
                noise = torch.randn(x.shape, generator=rng) * noise_std
                x = x + noise
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_corr += _pearson_corr(pred, y)
            n_batches += 1
    return total_corr / n_batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="reports/mmecg_comparison/noise_robustness.json")
    args = parser.parse_args()

    results = []
    for run_str in args.runs:
        run_dir = Path(run_str)
        config = load_config(run_dir / "config.yaml")
        data_cfg = config["data"]
        raw_root = REPO_ROOT / data_cfg["raw_root"]
        trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"]["test"])
        test_ds = MMECGWindowDataset(
            raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"],
            complex_input=data_cfg.get("complex_input", False),
        )
        loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = _load_model(run_dir, config)
        model_name = config["model"]["name"]

        curve = []
        for noise_std in NOISE_STDS:
            seeds = [0] if noise_std == 0.0 else list(range(N_SEEDS_PER_LEVEL))
            corrs = [_evaluate_at_noise_level(model, loader, noise_std, seed) for seed in seeds]
            mean_corr = float(np.mean(corrs))
            curve.append({"noise_std": noise_std, "corr": mean_corr})
            print(f"{model_name:15s} noise_std={noise_std:.2f} corr={mean_corr:.4f}")

        results.append({"run": run_dir.name, "model": model_name, "curve": curve})

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
