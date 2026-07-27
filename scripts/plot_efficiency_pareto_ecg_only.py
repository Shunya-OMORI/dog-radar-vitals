"""波形推定モデル(ecg_*系)だけに絞った「精度(波形相関) vs 効率性(パラメータ数)」Pareto図。

`plot_efficiency_pareto.py`（rpeak_*系も混在、精度軸はRR Interval MAE）とは異なり、
本スクリプトは(a) ecg_*系のみ、(b) 精度軸は`scripts/plot_waveform_comparison.py`が算出した
test被験者全体での正式な波形相関（`waveform_prediction_comparison.json`）を使う。
波形推定モデルにとっては、間接的なRR MAEよりも相関係数の方が直接的な「精度」の物差しになる
（ユーザとの会話「今回の問題だとAccuracyはどうやって定義されるの？」参照）。

使い方:
    python scripts/plot_efficiency_pareto_ecg_only.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = REPO_ROOT / "reports" / "mmecg_comparison"

# Okabe-Ito配色。plot_efficiency_pareto.pyと同じ規約を踏襲する。
FAMILY_COLORS = {
    "unet1d": "#56B4E9",
    "conformer": "#E69F00",
    "complex_cnn_v2": "#CC79A7",
    "complex_cnn": "#F0E442",
    "transformer": "#999999",
    "cnn1d": "#0072B2",
    "lstm": "#009E73",
    "conv_ncp": "#D55E00",
    "ncp": "#D55E00",
    "spatial_gnn": "#332288",
    "spatial_fusion": "#117733",
}


def _family_of(model_name: str) -> str:
    for key in FAMILY_COLORS:
        if key in model_name:
            return key
    return "cnn1d"


def main() -> None:
    efficiency = json.loads((COMPARISON_DIR / "efficiency_benchmark.json").read_text())
    correlation = json.loads((COMPARISON_DIR / "waveform_prediction_comparison.json").read_text())

    corr_by_model = {r["model"]: r["full_test_corr"] for r in correlation}
    eff_by_model = {e["model"]: e for e in efficiency}

    # ecg_*系のみ(waveform_prediction_comparison.jsonに載っている11モデル)。
    points = []
    for model_name, corr in corr_by_model.items():
        eff = eff_by_model.get(model_name)
        if eff is None:
            print(f"skip: no efficiency data for {model_name}")
            continue
        points.append({**eff, "corr": corr, "family": _family_of(model_name)})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 全点に直接ラベルを付けるため凡例は使わない(選択的直接ラベルの原則、dataviz skill参照)。
    for ax, (xkey, xlabel) in zip(axes, [("n_params", "Parameters"), ("cpu_latency_ms", "CPU Inference Latency [ms] (batch=1)")]):
        for p in points:
            color = FAMILY_COLORS[p["family"]]
            ax.scatter(p[xkey], p["corr"], color=color, s=70, zorder=3)
            ax.annotate(p["model"], (p[xkey], p["corr"]), fontsize=8, xytext=(5, 4), textcoords="offset points")

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Waveform Pearson correlation (higher is better)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", axis="both", alpha=0.2)
    fig.suptitle("Waveform Models Only (ecg_*): Accuracy vs. Efficiency Trade-off")
    fig.tight_layout()
    out_path = COMPARISON_DIR / "efficiency_pareto_ecg_only.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
