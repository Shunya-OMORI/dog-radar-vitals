"""犬10頭をn_folds個のfoldに分け、犬を入れ替えながら学習・評価するleave-few-dogs-out CV。

単一の固定train/val/test分割（configs/base.yaml）では、val 1頭・test 2頭という
特定の個体への適合/不適合がモデル比較の結論を左右しかねないことが分かっている
（EXPERIMENTS.md「HR Transformer対照実験の結果」参照）。このスクリプトは、
foldごとにtrain/val/test犬の組を入れ替えて学習・評価を繰り返し、結果が特定の
犬の組み合わせに依存していないかを確認する。

使い方:
    python scripts/run_dog_cross_validation.py \
        --config configs/experiments/013_transformer_hr_lr5e-4.yaml --n-folds 5 --tag cv_013
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.baselines import compute_baselines  # noqa: E402
from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.scenario1 import list_dogs  # noqa: E402
from dog_radar_vitals.evaluate import evaluate_run  # noqa: E402
from dog_radar_vitals.train import make_run_dir, train_from_config  # noqa: E402


def make_folds(dogs: list[str], n_folds: int, seed: int) -> list[list[str]]:
    """犬をシャッフルしてn_folds個のtest群に等分する（決定的、シード固定）。"""
    rng = np.random.RandomState(seed)
    shuffled = list(dogs)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def run_cross_validation(config_path: str, n_folds: int, tag: str) -> dict:
    base_config = load_config(config_path)
    raw_root = REPO_ROOT / base_config["data"]["raw_root"]
    all_dogs = list_dogs(raw_root)
    folds = make_folds(all_dogs, n_folds, seed=base_config["seed"])

    results = []
    for fold_idx, test_dogs in enumerate(folds):
        remaining = [d for d in all_dogs if d not in test_dogs]
        val_dogs = remaining[:1]
        train_dogs = remaining[1:]

        config = copy.deepcopy(base_config)
        config["data"]["dogs"] = {"train": train_dogs, "val": val_dogs, "test": test_dogs}

        run_dir = make_run_dir(config, tag=f"{tag}_fold{fold_idx}")
        print(f"=== fold {fold_idx}: train={train_dogs} val={val_dogs} test={test_dogs} -> {run_dir} ===")
        train_from_config(config, run_dir)
        test_metrics = evaluate_run(run_dir)

        primary_key = "mae" if "mae" in test_metrics else "corr"
        trivial_mae = None
        if primary_key == "mae":
            trivial_mae = compute_baselines(config["task"], raw_root, config["data"]["dogs"])["trivial_mae"]

        results.append(
            {
                "fold": fold_idx,
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "train_dogs": train_dogs,
                "val_dogs": val_dogs,
                "test_dogs": test_dogs,
                "test_metric_name": primary_key,
                "test_metric": test_metrics[primary_key],
                "trivial_mae": trivial_mae,
            }
        )

    values = [r["test_metric"] for r in results]
    return {
        "config": config_path,
        "n_folds": n_folds,
        "task": base_config["task"],
        "model": base_config["model"]["name"],
        "family": base_config["model"]["family"],
        "folds": results,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="train/val/test犬指定以外を流用するベースconfig")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    summary = run_cross_validation(args.config, args.n_folds, args.tag)

    manifest_dir = REPO_ROOT / "runs" / "_cross_validation"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.tag}.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(
        f"\n=== {summary['model']} ({summary['task']}): mean={summary['mean']:.3f} "
        f"std={summary['std']:.3f} across {summary['n_folds']} folds ==="
    )
    for r in summary["folds"]:
        print(f"  fold{r['fold']}: test_dogs={r['test_dogs']} {r['test_metric_name']}={r['test_metric']:.3f}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
