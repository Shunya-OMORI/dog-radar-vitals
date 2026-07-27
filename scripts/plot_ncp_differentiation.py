"""ノイズ頑健性実験・データ効率実験の結果を可視化する（NCPを別軸で評価する狙い、
EXPERIMENTS.md「NCPを差別化軸で評価する」参照）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = REPO_ROOT / "reports" / "mmecg_comparison"

# Okabe-Ito配色。make_comparison_report.py/plot_efficiency_pareto.pyと同じ規約。
MODEL_COLORS = {"ecg_ncp": "#D55E00", "ecg_cnn1d": "#0072B2", "ecg_unet1d": "#56B4E9"}
MODEL_LABELS = {"ecg_ncp": "NCP", "ecg_cnn1d": "CNN", "ecg_unet1d": "U-Net"}

# データ効率実験の結果（train被験者数ごとのtest相関）。evaluate.pyの出力を手動で集約。
DATA_EFFICIENCY = {
    "ecg_ncp": {2: -0.022, 4: 0.038, 7: 0.115},
    "ecg_cnn1d": {2: 0.080, 4: 0.184, 7: 0.224},
    "ecg_unet1d": {2: 0.013, 4: 0.104, 7: 0.209},
}


def plot_noise_robustness() -> None:
    data = json.loads((COMPARISON_DIR / "noise_robustness.json").read_text())

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for entry in data:
        model = entry["model"]
        xs = [p["noise_std"] for p in entry["curve"]]
        ys = [p["corr"] for p in entry["curve"]]
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=2)

    ax.set_xlabel("Injected noise std (relative to z-scored signal, std=1)")
    ax.set_ylabel("Test Pearson correlation")
    ax.set_title("Noise Robustness: NCP vs. CNN vs. U-Net")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = COMPARISON_DIR / "noise_robustness.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def plot_data_efficiency() -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model, curve in DATA_EFFICIENCY.items():
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model], linewidth=2)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of training subjects")
    ax.set_ylabel("Test Pearson correlation")
    ax.set_title("Data Efficiency: NCP vs. CNN vs. U-Net")
    ax.set_xticks([2, 4, 7])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path = COMPARISON_DIR / "data_efficiency.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    plot_noise_robustness()
    plot_data_efficiency()
