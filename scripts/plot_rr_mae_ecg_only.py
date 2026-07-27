"""波形推定モデル(ecg_*系)だけに絞ったRR Interval MAEランキング。

`scripts/compare_mmecg_models.py`の出力はrpeak_*系(heatmap回帰)も混在しており、
数値がRR Intervalの検出精度で上位を占めるため、ecg_*系(密な波形回帰)だけを取り出して
比較したい、というユーザの要望に応える。各runのcompare結果(`reports/mmecg_comparison/*.json`
および実行時の一時json)からecg_*系のRR MAE・F1を集約する。

使い方:
    python scripts/plot_rr_mae_ecg_only.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# 201-235フェーズの各compare実行結果から集約したecg_*系のRR Interval MAE・F1。
# 出典: reports/mmecg_comparison/comparison_full.json (201-215本体),
#       reports/mmecg_comparison/comparison_v2_partial.json (216-222本体),
#       conv_ncp_compare.json (233), spatial_gnn_compare.json (235),
#       spatial_compare.json (231)。EXPERIMENTS.md本文の数値と一致。
ECG_RR_MAE = {
    "ecg_unet1d": (15.9, 0.498),
    "ecg_spatial_fusion": (16.7, 0.552),
    "ecg_spatial_gnn": (17.95, 0.492),
    "ecg_conv_ncp": (19.47, 0.400),
    "ecg_complex_cnn": (19.4, 0.373),
    "ecg_ncp": (21.1, 0.311),
    "ecg_cnn1d": (21.2, 0.479),
    "ecg_lstm": (21.5, 0.350),
    "ecg_conformer": (21.8, 0.409),
    "ecg_complex_cnn_v2": (22.1, 0.447),
    "ecg_transformer": (23.5, 0.436),
}

PRED_COLOR = "#0072B2"


def main() -> None:
    items = sorted(ECG_RR_MAE.items(), key=lambda kv: kv[1][0])
    names = [k for k, _ in items]
    maes = [v[0] for _, v in items]
    f1s = [v[1] for _, v in items]

    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(items) + 1))
    bars = ax.barh(names, maes, color=PRED_COLOR, height=0.6)
    for bar, mae, f1 in zip(bars, maes, f1s):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{mae:.1f}ms (F1={f1:.2f})", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("RR Interval MAE [ms] (lower is better)")
    ax.set_title("Waveform Models Only (ecg_*): RR Interval MAE Ranking\n(rpeak_* heatmap models excluded)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(maes) * 1.2)
    fig.tight_layout()

    out_path = REPO_ROOT / "reports" / "mmecg_comparison" / "rr_mae_ecg_only.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
