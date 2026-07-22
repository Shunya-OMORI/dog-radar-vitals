"""101(波形回帰)と102(heatmap回帰)を、被験者を入れ替えたn-fold cross-validationで比較する。

犬データの`run_dog_cross_validation.py`と同じ設計思想: 単発の1分割（test 2名）だけでは
「どちらが優れているか」を結論づけられないため、被験者をシャッフルしてfoldごとに
train/val/testを入れ替え、両モデルを同一のfold構成で学習・評価する。

評価は`rpeak_evaluation.py`のR波検出ベースの指標（F1・RR Interval MAE）で揃え、
foldをまたいで得られる(ecg_cnn1d, rpeak_cnn1d)の対応のあるペアに対して
Wilcoxon符号順位検定を行う。

使い方:
    python scripts/run_ecg_cross_validation.py \
        --ecg-config configs/experiments/101_ecg_cnn1d_resting.yaml \
        --rpeak-config configs/experiments/102_rpeak_cnn1d_resting.yaml \
        --n-folds 5 --tag ecg_cv001
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.schellenberger import list_subjects  # noqa: E402
from dog_radar_vitals.rpeak_evaluation import evaluate_peak_detection, load_ecg_model  # noqa: E402
from dog_radar_vitals.train import make_run_dir, train_from_config  # noqa: E402


def make_folds(subjects: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]


def _train_one(config_path: str, subjects_split: dict, tag: str) -> Path:
    config = load_config(config_path)
    config["data"]["subjects"] = subjects_split
    run_dir = make_run_dir(config, tag=tag)
    train_from_config(config, run_dir)
    return run_dir


def run_cross_validation(ecg_config_path: str, rpeak_config_path: str, n_folds: int, tag: str) -> dict:
    base_config = load_config(ecg_config_path)
    raw_root = REPO_ROOT / base_config["data"]["raw_root"]
    scenario = base_config["data"]["scenario"]
    window_sec = base_config["data"]["window_sec"]

    all_subjects = list_subjects(raw_root)
    folds = make_folds(all_subjects, n_folds, seed=base_config["seed"])

    per_subject_results = []
    for fold_idx, test_subjects in enumerate(folds):
        remaining = [s for s in all_subjects if s not in test_subjects]
        val_subjects = remaining[:1]
        train_subjects = remaining[1:]
        split = {"train": train_subjects, "val": val_subjects, "test": test_subjects}

        print(f"=== fold {fold_idx}: train={len(train_subjects)} val={val_subjects} test={test_subjects} ===")

        ecg_run_dir = _train_one(ecg_config_path, split, tag=f"{tag}_ecg_fold{fold_idx}")
        rpeak_run_dir = _train_one(rpeak_config_path, split, tag=f"{tag}_rpeak_fold{fold_idx}")

        ecg_config = load_config(ecg_run_dir / "config.yaml")
        rpeak_config = load_config(rpeak_run_dir / "config.yaml")
        ecg_model = load_ecg_model(ecg_run_dir, ecg_config)
        rpeak_model = load_ecg_model(rpeak_run_dir, rpeak_config)

        for subject_id in test_subjects:
            ecg_result = evaluate_peak_detection(ecg_model, "waveform", subject_id, raw_root, scenario, window_sec)
            rpeak_result = evaluate_peak_detection(rpeak_model, "heatmap", subject_id, raw_root, scenario, window_sec)
            per_subject_results.append(
                {
                    "fold": fold_idx,
                    "subject": subject_id,
                    "ecg_run_dir": str(ecg_run_dir.relative_to(REPO_ROOT)),
                    "rpeak_run_dir": str(rpeak_run_dir.relative_to(REPO_ROOT)),
                    "ecg_cnn1d": ecg_result,
                    "rpeak_cnn1d": rpeak_result,
                }
            )
            print(
                f"  {subject_id}: ecg_cnn1d f1={ecg_result['f1']:.3f} rr_mae={ecg_result['rr_mae_ms']} | "
                f"rpeak_cnn1d f1={rpeak_result['f1']:.3f} rr_mae={rpeak_result['rr_mae_ms']}"
            )

    return {
        "ecg_config": ecg_config_path,
        "rpeak_config": rpeak_config_path,
        "n_folds": n_folds,
        "n_subjects": len(all_subjects),
        "per_subject": per_subject_results,
    }


def summarize(result: dict) -> dict:
    f1_ecg = np.array([r["ecg_cnn1d"]["f1"] for r in result["per_subject"]])
    f1_rpeak = np.array([r["rpeak_cnn1d"]["f1"] for r in result["per_subject"]])

    rr_ecg = [r["ecg_cnn1d"]["rr_mae_ms"] for r in result["per_subject"]]
    rr_rpeak = [r["rpeak_cnn1d"]["rr_mae_ms"] for r in result["per_subject"]]
    rr_pairs = [(a, b) for a, b in zip(rr_ecg, rr_rpeak) if a is not None and b is not None]

    f1_diff = f1_rpeak - f1_ecg  # 正ならheatmap回帰が良い
    f1_wilcoxon = None
    if np.count_nonzero(f1_diff) >= 2:
        w_stat, w_p = stats.wilcoxon(f1_diff)
        f1_wilcoxon = {"statistic": float(w_stat), "p_value": float(w_p)}

    rr_wilcoxon = None
    if len(rr_pairs) >= 2:
        rr_diff = np.array([b - a for a, b in rr_pairs])  # 正ならheatmap回帰がRR誤差大(悪い)
        if np.count_nonzero(rr_diff) >= 2:
            w_stat, w_p = stats.wilcoxon(rr_diff)
            rr_wilcoxon = {"statistic": float(w_stat), "p_value": float(w_p)}

    return {
        "n_subjects_evaluated": len(result["per_subject"]),
        "f1_mean": {"ecg_cnn1d": float(f1_ecg.mean()), "rpeak_cnn1d": float(f1_rpeak.mean())},
        "f1_std": {"ecg_cnn1d": float(f1_ecg.std()), "rpeak_cnn1d": float(f1_rpeak.std())},
        "f1_wilcoxon_p_value": f1_wilcoxon["p_value"] if f1_wilcoxon else None,
        "n_rr_pairs": len(rr_pairs),
        "rr_mae_mean": {
            "ecg_cnn1d": float(np.mean([a for a, _ in rr_pairs])) if rr_pairs else None,
            "rpeak_cnn1d": float(np.mean([b for _, b in rr_pairs])) if rr_pairs else None,
        },
        "rr_wilcoxon_p_value": rr_wilcoxon["p_value"] if rr_wilcoxon else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg-config", required=True)
    parser.add_argument("--rpeak-config", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    result = run_cross_validation(args.ecg_config, args.rpeak_config, args.n_folds, args.tag)
    summary = summarize(result)
    result["summary"] = summary

    manifest_dir = REPO_ROOT / "runs" / "_cross_validation"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.tag}.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n=== summary (n_subjects_evaluated={summary['n_subjects_evaluated']}) ===")
    print(f"F1: ecg_cnn1d={summary['f1_mean']['ecg_cnn1d']:.3f}±{summary['f1_std']['ecg_cnn1d']:.3f}  "
          f"rpeak_cnn1d={summary['f1_mean']['rpeak_cnn1d']:.3f}±{summary['f1_std']['rpeak_cnn1d']:.3f}  "
          f"Wilcoxon p={summary['f1_wilcoxon_p_value']}")
    if summary["rr_mae_mean"]["ecg_cnn1d"] is not None:
        print(f"RR MAE(matched, n={summary['n_rr_pairs']}): ecg_cnn1d={summary['rr_mae_mean']['ecg_cnn1d']:.1f}ms  "
              f"rpeak_cnn1d={summary['rr_mae_mean']['rpeak_cnn1d']:.1f}ms  Wilcoxon p={summary['rr_wilcoxon_p_value']}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
