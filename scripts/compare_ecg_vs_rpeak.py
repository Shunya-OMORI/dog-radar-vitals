"""波形回帰(101 ecg_cnn1d)とheatmap回帰(102 rpeak_cnn1d)を、単発の学習run同士で比較する。

評価ロジック本体は`dog_radar_vitals.rpeak_evaluation`にある（cross-validation版の
`run_ecg_cross_validation.py`と共有するため）。

使い方:
    python scripts/compare_ecg_vs_rpeak.py --ecg-run runs/xxx_ecg_ecg_cnn1d --rpeak-run runs/yyy_ecg_rpeak_cnn1d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.rpeak_evaluation import evaluate_peak_detection, load_ecg_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg-run", required=True, help="101(ecg_cnn1d)のrun_dir")
    parser.add_argument("--rpeak-run", required=True, help="102(rpeak_cnn1d)のrun_dir")
    parser.add_argument("--out", default=None, help="結果jsonの保存先")
    args = parser.parse_args()

    ecg_run_dir = Path(args.ecg_run)
    rpeak_run_dir = Path(args.rpeak_run)
    ecg_config = load_config(ecg_run_dir / "config.yaml")
    rpeak_config = load_config(rpeak_run_dir / "config.yaml")

    raw_root = REPO_ROOT / ecg_config["data"]["raw_root"]
    scenario = ecg_config["data"]["scenario"]
    window_sec = ecg_config["data"]["window_sec"]
    test_subjects = ecg_config["data"]["subjects"]["test"]

    ecg_model = load_ecg_model(ecg_run_dir, ecg_config)
    rpeak_model = load_ecg_model(rpeak_run_dir, rpeak_config)

    results = []
    for subject_id in test_subjects:
        ecg_result = evaluate_peak_detection(ecg_model, "waveform", subject_id, raw_root, scenario, window_sec)
        rpeak_result = evaluate_peak_detection(rpeak_model, "heatmap", subject_id, raw_root, scenario, window_sec)
        results.append({"subject": subject_id, "n_true_peaks": ecg_result["n_true_peaks"], "ecg_cnn1d": ecg_result, "rpeak_cnn1d": rpeak_result})

        print(f"=== {subject_id} (n_true_peaks={ecg_result['n_true_peaks']}) ===")
        for model_name, r in [("ecg_cnn1d", ecg_result), ("rpeak_cnn1d", rpeak_result)]:
            rr_mae_str = f"{r['rr_mae_ms']:.1f}ms" if r["rr_mae_ms"] is not None else "N/A"
            print(f"  {model_name}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f} rr_mae={rr_mae_str}")

    out_path = Path(args.out) if args.out else REPO_ROOT / "reports" / "20260723_ecg_vs_rpeak" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
