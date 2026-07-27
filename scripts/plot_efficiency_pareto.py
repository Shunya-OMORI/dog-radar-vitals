"""benchmark_model_efficiency.pyとcompare_mmecg_models.pyの出力を突き合わせ、
「精度(RR Interval MAE) vs パラメータ数/CPU推論レイテンシ」のPareto図を作る。

NCPが「パラメータ数・FLOPsは桁違いに少ないが、CfCの逐次展開ゆえCPU推論レイテンシは
逆に最も遅い」という、精度単体の比較表だけでは見えないトレードオフを可視化する
（`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「NCP/LTCの効率性評価」参照）。

使い方:
    python scripts/plot_efficiency_pareto.py \
        --efficiency reports/mmecg_comparison/efficiency_benchmark.json \
        --comparison reports/mmecg_comparison/comparison_full.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito配色。make_comparison_report.pyと同じ規約を踏襲し、モデル系統ごとに固定色を割り当てる。
# 判定順序が重要（"complex_cnn_v2"が"complex_cnn"に先にマッチしないよう、より具体的な
# キーを先に置く）。
FAMILY_COLORS = {
    "unet1d": "#56B4E9",
    "conformer": "#E69F00",
    "complex_cnn_v2": "#CC79A7",
    "complex_cnn": "#F0E442",
    "transformer": "#999999",
    "cnn1d": "#0072B2",
    "lstm": "#009E73",
    "ncp": "#D55E00",
}


def _family_of(model_name: str) -> str:
    for key in FAMILY_COLORS:
        if key in model_name:
            return key
    return "cnn1d"


def make_pareto_chart(efficiency: list[dict], rr_table: list[dict], out_path: Path) -> None:
    rr_by_model = {r["model"]: r["rr_mae_ms_mean"] for r in rr_table if r.get("rr_mae_ms_mean") is not None}

    points = []
    for e in efficiency:
        mae = rr_by_model.get(e["model"])
        if mae is None:
            continue
        points.append({**e, "rr_mae_ms": mae, "family": _family_of(e["model"])})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (xkey, xlabel, xlog) in zip(
        axes,
        [("n_params", "Parameters", True), ("cpu_latency_ms", "CPU Inference Latency [ms] (batch=1)", True)],
    ):
        seen_families = set()
        for p in points:
            color = FAMILY_COLORS[p["family"]]
            label = p["family"] if p["family"] not in seen_families else None
            seen_families.add(p["family"])
            ax.scatter(p[xkey], p["rr_mae_ms"], color=color, s=70, label=label, zorder=3)
            ax.annotate(p["model"], (p[xkey], p["rr_mae_ms"]), fontsize=8, xytext=(5, 4), textcoords="offset points")

        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RR Interval MAE [ms] (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, which="both", axis="both", alpha=0.2)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Accuracy vs. Efficiency Trade-off (MMECG, single split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--efficiency", default="reports/mmecg_comparison/efficiency_benchmark.json")
    parser.add_argument("--comparison", default="reports/mmecg_comparison/comparison_full.json")
    parser.add_argument("--out", default="reports/mmecg_comparison/efficiency_pareto.png")
    args = parser.parse_args()

    efficiency = json.loads((REPO_ROOT / args.efficiency).read_text())
    comparison = json.loads((REPO_ROOT / args.comparison).read_text())
    make_pareto_chart(efficiency, comparison["rr_interval"], REPO_ROOT / args.out)


if __name__ == "__main__":
    main()
