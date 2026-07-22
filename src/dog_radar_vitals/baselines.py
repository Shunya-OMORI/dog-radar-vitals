"""モデルのMAEを評価する際の物差しとなる2つのベースラインを計算する。

- trivial_mae: train犬の目的変数の平均値を、test窓すべてに対して常に予測した場合のMAE。
  「レーダから何も学習していない」場合に相当する下限の目安であり、モデルのMAEはこれより
  十分小さくなければ意味がない。
- oracle_mae: 各test犬の「その犬自身の真の平均値」を予測できたと仮定した場合のMAE。
  犬間の個体差を除いた、窓内変動のみに起因する理論的な下限（=これより下げるには
  個体を特定するのと事実上同じ情報が要る）。

`scripts/compute_baselines.py`（固定分割用CLI）と
`scripts/run_dog_cross_validation.py`（fold毎の分割用）の両方から使う共通ロジック。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dog_radar_vitals.data.features import build_feature_table


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
