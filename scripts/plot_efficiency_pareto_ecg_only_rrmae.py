"""波形推定モデル(ecg_*系)だけに絞った「RR Interval MAE vs 効率性」Pareto図。

`plot_efficiency_pareto.py`と同じ2パネル構成(横軸=パラメータ数/CPU推論レイテンシ)だが、
(a) ecg_*系のみ、(b) `plot_efficiency_pareto_ecg_only.py`（精度軸=波形相関）とは異なり、
精度軸はRR Interval MAE[ms]（低いほど良い）を使う。データは`plot_rr_mae_ecg_only.py`の
`ECG_RR_MAE`と`efficiency_benchmark.json`を突き合わせる。

使い方:
    python scripts/plot_efficiency_pareto_ecg_only_rrmae.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = REPO_ROOT / "reports" / "mmecg_comparison"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_rr_mae_ecg_only import ECG_RR_MAE  # noqa: E402

# Okabe-Ito配色。plot_efficiency_pareto_ecg_only.pyと同じ規約。
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
    eff_by_model = {e["model"]: e for e in efficiency}

    points = []
    for model_name, (rr_mae, f1) in ECG_RR_MAE.items():
        eff = eff_by_model.get(model_name)
        if eff is None:
            print(f"skip: no efficiency data for {model_name}")
            continue
        points.append({**eff, "rr_mae": rr_mae, "f1": f1, "family": _family_of(model_name)})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ecg_conv_ncpとecg_complex_cnnはRR MAEがほぼ同値(19.4-19.5ms)でラベルが重なるため、
    # 片方だけ下にオフセットする。
    label_offsets = {"ecg_conv_ncp": (5, -10)}

    for ax, (xkey, xlabel) in zip(axes, [("n_params", "Parameters"), ("cpu_latency_ms", "CPU Inference Latency [ms] (batch=1)")]):
        for p in points:
            color = FAMILY_COLORS[p["family"]]
            ax.scatter(p[xkey], p["rr_mae"], color=color, s=70, zorder=3)
            offset = label_offsets.get(p["model"], (5, 4))
            ax.annotate(p["model"], (p[xkey], p["rr_mae"]), fontsize=8, xytext=offset, textcoords="offset points")

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RR Interval MAE [ms] (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", axis="both", alpha=0.2)

    fig.suptitle("Waveform Models Only (ecg_*): RR Interval MAE vs. Efficiency Trade-off")
    fig.tight_layout()
    out_path = COMPARISON_DIR / "efficiency_pareto_ecg_only_rrmae.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
