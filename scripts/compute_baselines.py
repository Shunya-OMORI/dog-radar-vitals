"""固定のtrain/val/test犬分割（configs/base.yaml）に対するtrivial/oracleベースラインを計算する。

計算ロジック本体は `dog_radar_vitals.baselines` にある（fold毎に計算し直す
`scripts/run_dog_cross_validation.py` と共有するため）。

使い方:
    python scripts/compute_baselines.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.baselines import compute_baselines  # noqa: E402
from dog_radar_vitals.config import load_config  # noqa: E402


def main() -> None:
    config = load_config(REPO_ROOT / "configs" / "base.yaml")
    raw_root = REPO_ROOT / config["data"]["raw_root"]
    dogs = config["data"]["dogs"]

    lines = [
        "# ベースライン（モデルのMAEを評価する物差し）",
        "",
        "- **trivial_mae**: train犬の平均値を常に予測した場合のMAE。モデルはこれより十分小さくなければ",
        "  「レーダから何も学習していない」のと変わらない。",
        "- **oracle_mae**: 各test犬の真の平均値を知っていたと仮定した場合のMAE。個体差を除いた",
        "  窓内変動のみに起因する理論的な下限。",
        "",
        "| task | train_mean | test_range | trivial_mae | oracle_mae |",
        "|---|---|---|---|---|",
    ]
    for task in ["hr", "br"]:
        r = compute_baselines(task, raw_root, dogs)
        lines.append(
            f"| {r['task']} | {r['train_mean']:.2f} | [{r['test_min']:.1f}, {r['test_max']:.1f}] | "
            f"{r['trivial_mae']:.3f} | {r['oracle_mae']:.3f} |"
        )
        print(f"{task}: trivial_mae={r['trivial_mae']:.3f} oracle_mae={r['oracle_mae']:.3f}")

    out_path = REPO_ROOT / "reports" / "baselines.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
