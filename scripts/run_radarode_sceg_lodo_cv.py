"""config269(採用構成)を11名全員でLeave-One-Subject-Out CVにかけ、test_corrのばらつきを見る。

進捗報告書 Part III-9 の残課題3「学習7名・検証1名では検証値がepoch間で±0.08揺れる。
先行研究と同じ11-fold交差検証に切り替える」に対応する。**モデル構成は269から一切変えない
(R4: 単一変数)。変えるのは被験者分割だけ。**

各fold: 9名train・1名val(best epoch選択用)・1名test。11名なので11fold全部で
ちょうど全員が一度ずつtestに回る(Leave-One-Subject-Out)。

使い方:
    .venv/bin/python scripts/run_radarode_sceg_lodo_cv.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.train import make_run_dir  # noqa: E402
from dog_radar_vitals.training.radarode_sceg_trainer import (  # noqa: E402
    evaluate_radarode_sceg,
    train_radarode_sceg,
)

BASE_CONFIG_PATH = REPO_ROOT / "configs/experiments/269_mmecg_radarode_sceg_hires.yaml"
ALL_SUBJECTS = [1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30]
OUTPUT_PATH = REPO_ROOT / "reports/radarode_sceg_lodo_cv.json"


def main() -> None:
    base_config = load_config(str(BASE_CONFIG_PATH))
    results = []

    for i, test_subject in enumerate(ALL_SUBJECTS):
        remaining = [s for s in ALL_SUBJECTS if s != test_subject]
        val_subject = remaining[i % len(remaining)]
        train_subjects = [s for s in remaining if s != val_subject]

        config = json.loads(json.dumps(base_config))  # deep copy
        config["data"]["subjects"] = {
            "train": train_subjects,
            "val": [val_subject],
            "test": [test_subject],
        }

        tag = f"lodo269_fold{i}_test{test_subject}"
        run_dir = make_run_dir(config, tag=tag)
        print(f"\n=== fold {i}: test_subject={test_subject} val_subject={val_subject} ===")
        print(f"train_subjects={train_subjects}")
        train_radarode_sceg(config, run_dir, REPO_ROOT)
        metrics = evaluate_radarode_sceg(run_dir, config, REPO_ROOT)
        print(f"fold {i} test_corr={metrics['corr']:.3f}")

        results.append(
            {
                "fold": i,
                "test_subject": test_subject,
                "val_subject": val_subject,
                "train_subjects": train_subjects,
                "test_corr": metrics["corr"],
                "test_loss": metrics["loss"],
                "run_dir": str(run_dir),
            }
        )
        OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    corrs = [r["test_corr"] for r in results]
    mean_corr = sum(corrs) / len(corrs)
    std_corr = (sum((c - mean_corr) ** 2 for c in corrs) / len(corrs)) ** 0.5
    summary = {"n_folds": len(results), "mean_test_corr": mean_corr, "std_test_corr": std_corr, "folds": results}
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== LODO CV summary: mean_test_corr={mean_corr:.3f} std={std_corr:.3f} over {len(results)} folds ===")
    print(f"(単一split269の元のtest_corr=0.300は3名test平均。今回は1名ずつ×11foldの平均・分散を見る)")


if __name__ == "__main__":
    main()
