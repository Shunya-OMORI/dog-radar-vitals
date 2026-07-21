"""古典MLモデル(scikit-learn)の学習。deep_trainerと同じmetrics.json形式で記録し、
比較レポートが両方を同一に扱えるようにする。
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from dog_radar_vitals.data.features import build_feature_table
from dog_radar_vitals.models.classical.registry import build_classical_model
from dog_radar_vitals.seeding import set_all_seeds


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def train_classical(config: dict, run_dir: Path, repo_root: Path) -> None:
    set_all_seeds(config["seed"])

    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    task = config["task"]

    X_train, y_train = build_feature_table(
        raw_root, data_cfg["dogs"]["train"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    X_val, y_val = build_feature_table(
        raw_root, data_cfg["dogs"]["val"], task, data_cfg["window_sec"], data_cfg["stride_sec"]
    )

    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_classical_model(config["model"]["name"], **model_kwargs)
    model.fit(X_train, y_train)

    train_mae = _mae(y_train, model.predict(X_train))
    val_mae = _mae(y_val, model.predict(X_val))
    print(f"train_mae={train_mae:.3f} val_mae={val_mae:.3f}")

    joblib.dump(model, run_dir / "best_model.joblib")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"history": [{"epoch": 0, "train": {"mae": train_mae}, "val": {"mae": val_mae}}], "best_val_mae": val_mae},
            indent=2,
        )
    )


def evaluate_classical(run_dir: Path, config: dict, repo_root: Path) -> dict[str, float]:
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]

    X_test, y_test = build_feature_table(
        raw_root, data_cfg["dogs"]["test"], config["task"], data_cfg["window_sec"], data_cfg["stride_sec"]
    )
    model = joblib.load(run_dir / "best_model.joblib")
    test_mae = _mae(y_test, model.predict(X_test))
    return {"mae": test_mae}
