"""あるrunが「個体内の時間変動」を実際に追えているかを診断する。

trivialベースラインを上回るMAEは、真に個体内の生理信号を捉えている場合だけでなく、
単に「母集団平均に近い値を返す」ことでも(運が良ければ)達成しうる。後者は健康
モニタリング（個体内の変化検知）には使えない。この診断は、test犬ごとに
「真値の時間変動」と「予測値の時間変動」の相関(within-dog correlation)を計算し、
オフセット当てゲームになっていないかを検証する。

使い方:
    python scripts/diagnose_within_dog_signal.py --run runs/xxx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.dataset import WindowedVitalsDataset  # noqa: E402
from dog_radar_vitals.models.deep.registry import build_deep_model  # noqa: E402


def diagnose(run_dir: Path) -> list[dict]:
    config = load_config(run_dir / "config.yaml")
    if config["model"]["family"] != "deep":
        raise ValueError("この診断は深層モデル(family=deep)のみ対応")

    raw_root = REPO_ROOT / config["data"]["raw_root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_deep_model(config["model"]["name"], n_bins=467, **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()

    results = []
    for dog in config["data"]["dogs"]["test"]:
        ds = WindowedVitalsDataset(raw_root, [dog], config["task"], config["data"]["window_sec"], config["data"]["stride_sec"])
        xs = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
        ys = torch.stack([ds[i][1] for i in range(len(ds))]).numpy()
        with torch.no_grad():
            preds = model(xs).cpu().numpy().squeeze(-1)

        true_std = float(ys.std())
        corr = float(np.corrcoef(ys - ys.mean(), preds - preds.mean())[0, 1]) if true_std > 1e-6 else float("nan")
        results.append(
            {
                "dog": dog,
                "true_mean": float(ys.mean()),
                "true_std": true_std,
                "pred_mean": float(preds.mean()),
                "pred_std": float(preds.std()),
                "bias": float(preds.mean() - ys.mean()),
                "within_dog_corr": corr,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    results = diagnose(Path(args.run))
    for r in results:
        print(
            f"{r['dog']}: true_mean={r['true_mean']:.2f} true_std={r['true_std']:.3f}  "
            f"pred_mean={r['pred_mean']:.2f} pred_std={r['pred_std']:.3f}  "
            f"bias={r['bias']:+.2f}  within_dog_corr={r['within_dog_corr']:.3f}"
        )

    mean_corr = np.mean([r["within_dog_corr"] for r in results])
    print(f"\nmean within_dog_corr = {mean_corr:.3f}")
    if abs(mean_corr) < 0.3:
        print("-> 個体内の時間変動をほぼ追えていない。予測はオフセット当てに近い可能性が高い。")


if __name__ == "__main__":
    main()
