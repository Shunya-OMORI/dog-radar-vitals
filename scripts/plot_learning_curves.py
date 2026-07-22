"""複数runのval_mae学習曲線を重ねて描く。

収束の速さ・有無を比較するための専用ツール。make_comparison_report.py の棒グラフは
最終test_maeしか見せないため、「エポックを重ねれば収束するのか」を見るには
学習曲線そのものが要る。

使い方:
    python scripts/plot_learning_curves.py \
        --runs runs/xxx_lr1e-4=baseline(lr=1e-4) runs/yyy_lr5e-4=lr=5e-4 \
        --out reports/20260722_hr_transformer_lr_ablation/learning_curves.png \
        --title "Transformer HR: learning rate ablation"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_run_spec(spec: str) -> tuple[Path, str]:
    run_dir, _, label = spec.partition("=")
    if not label:
        label = run_dir
    return REPO_ROOT / run_dir, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="'run_dir=label' の形式で複数指定（例: runs/xxx=baseline runs/yyy=lr5e-4）",
    )
    parser.add_argument("--out", required=True, help="出力PNGのパス")
    parser.add_argument("--title", default="Learning curves")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for spec in args.runs:
        run_dir, label = _parse_run_spec(spec)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        epochs = [h["epoch"] for h in metrics["history"]]
        val_mae = [h["val"]["mae"] for h in metrics["history"]]
        ax.plot(epochs, val_mae, label=label, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MAE")
    ax.set_title(args.title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"chart: {out_path}")


if __name__ == "__main__":
    main()
