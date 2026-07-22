"""モデルのMAEを評価する際の物差しとなる2つのベースラインを計算する。

- trivial baseline: train犬の目的変数の平均値を、test窓すべてに対して常に予測した場合のMAE。
  「レーダから何も学習していない」場合に相当する下限の目安であり、モデルのMAEはこれより
  十分小さくなければ意味がない。
- oracle baseline: 各test犬の「その犬自身の真の平均値」を予測できたと仮定した場合のMAE。
  犬間の個体差を除いた、窓内変動のみに起因する理論的な下限（=これより下げるには
  個体を特定するのと事実上同じ情報が要る）。

使い方:
    python scripts/compute_baselines.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.features import build_feature_table  # noqa: E402


def compute_baselines(task: str, raw_root: Path, dogs: dict) -> dict:
    _, y_train = build_feature_table(raw_root, dogs["train"], task, 10, 1)
    _, y_test = build_feature_table(raw_root, dogs["test"], task, 10, 1)

    train_mean = float(y_train.mean())
    trivial_mae = float(np.abs(y_test - train_mean).mean())

    oracle_abs_err_sum, n = 0.0, 0
    for dog in dogs["test"]:
        _, y_dog = build_feature_table(raw_root, [dog], task, 10, 1)
        oracle_abs_err_sum += float(np.abs(y_dog - y_dog.mean()).sum())
        n += len(y_dog)
    oracle_mae = oracle_abs_err_sum / n

    return {
        "task": task,
        "train_mean": train_mean,
        "test_min": float(y_test.min()),
        "test_max": float(y_test.max()),
        "trivial_mae": trivial_mae,
        "oracle_mae": oracle_mae,
    }


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
