"""cross-validationのmanifestを読み、モデルがtrivialベースラインに対して統計的に有意な差を
持つかを検定する。

`run_dog_cross_validation.py`はfold平均・std・「trivialを上回ったfold数」までは出すが、
n=5〜10という少ないfold数では平均を眺めるだけでは偶然かどうか判断できない。連続量の
回帰誤差にはチャンスレベルという明確な下限が無いため、「trivialより誤差が小さいか」を
foldごとに対応させたWilcoxon符号順位検定（対応のあるノンパラメトリック検定）で確認する。

使い方:
    python scripts/test_cv_significance.py --manifest runs/_cross_validation/cv10_013_transformer_hr.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def test_significance(manifest: dict) -> dict:
    folds = manifest["folds"]
    if folds[0]["test_metric_name"] != "mae" or folds[0]["trivial_mae"] is None:
        raise ValueError("この検定はfamily=deep/classical（trivial_maeが計算済み）のみ対応")

    model_errs = np.array([f["test_metric"] for f in folds])
    trivial_errs = np.array([f["trivial_mae"] for f in folds])
    diff = model_errs - trivial_errs  # 負ならモデルがtrivialより良い

    n_beats_trivial = int((diff < 0).sum())
    n_folds = len(folds)

    # 符号検定（二項検定）: 「モデルがtrivialを上回る確率は1/2」という帰無仮説に対する両側検定
    sign_test = stats.binomtest(n_beats_trivial, n_folds, p=0.5, alternative="two-sided")

    # Wilcoxon符号順位検定: 差の大きさも考慮した対応のあるノンパラメトリック検定
    # 全foldで差がゼロだと計算できないため、非ゼロの差が2件未満なら実行しない
    wilcoxon_result = None
    if np.count_nonzero(diff) >= 2:
        w_stat, w_p = stats.wilcoxon(diff, alternative="less")  # モデル誤差 < trivial誤差 を検定
        wilcoxon_result = {"statistic": float(w_stat), "p_value": float(w_p)}

    return {
        "model": manifest["model"],
        "task": manifest["task"],
        "n_folds": n_folds,
        "n_beats_trivial": n_beats_trivial,
        "mean_diff": float(diff.mean()),  # 負ならモデルが平均的にtrivialより良い
        "sign_test_p_value": float(sign_test.pvalue),
        "wilcoxon": wilcoxon_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="run_dog_cross_validation.pyが出したjson")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    result = test_significance(manifest)

    print(f"model={result['model']} task={result['task']} n_folds={result['n_folds']}")
    print(f"  trivialを上回ったfold数: {result['n_beats_trivial']}/{result['n_folds']}")
    print(f"  平均差(model - trivial): {result['mean_diff']:+.3f}  (負ならモデルが平均的に良い)")
    print(f"  符号検定 p値: {result['sign_test_p_value']:.4f}")
    if result["wilcoxon"] is not None:
        print(f"  Wilcoxon符号順位検定 p値(片側, モデル<trivial): {result['wilcoxon']['p_value']:.4f}")
    else:
        print("  Wilcoxon符号順位検定: 非ゼロの差が2件未満のため実行不可")

    alpha = 0.05
    if result["sign_test_p_value"] < alpha or (result["wilcoxon"] and result["wilcoxon"]["p_value"] < alpha):
        print(f"  => 有意水準{alpha}で、trivialとの差は統計的に有意")
    else:
        print(f"  => 有意水準{alpha}で、trivialとの差は統計的に有意とは言えない")


if __name__ == "__main__":
    main()
