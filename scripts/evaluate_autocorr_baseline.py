"""古典的自己相関ベースラインを、深層学習モデルと同じevaluate_trial_peak_detection経路で
評価する(公平な比較のため、F1・RR-MAEの計算コードは共通化する)。CPUのみ・GPU不要。
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

import numpy as np

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects
from dog_radar_vitals.data.rpeaks import detect_r_peaks, match_peaks, matched_rr_interval_mae_ms
from dog_radar_vitals.models.classical.autocorr_rri import detect_peaks_autocorr

RAW_ROOT = "data/raw"
TEST_SUBJECTS = [17, 29, 30]  # config284と同一のtest split


def main() -> None:
    trial_ids = trial_ids_for_subjects(RAW_ROOT, TEST_SUBJECTS)
    results = []
    for trial_id in trial_ids:
        rec = load_trial(RAW_ROOT, trial_id)
        ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
        true_peaks = detect_r_peaks(ecg_filled, rec.fs)

        pred_peaks = detect_peaks_autocorr(rec.rcg, rec.fs)
        match = match_peaks(true_peaks, pred_peaks, rec.fs, tolerance_ms=50)
        rr_mae = matched_rr_interval_mae_ms(match, rec.fs)
        results.append({
            "trial_id": trial_id, "n_true_peaks": int(len(true_peaks)),
            "precision": match["precision"], "recall": match["recall"], "f1": match["f1"],
            "rr_mae_ms": rr_mae,
        })
        print(f"trial {trial_id}: f1={match['f1']:.3f} precision={match['precision']:.3f} recall={match['recall']:.3f} rr_mae_ms={rr_mae}")

    f1s = [r["f1"] for r in results]
    rr = [r["rr_mae_ms"] for r in results if r["rr_mae_ms"] is not None]
    print(f"\n[古典自己相関ベースライン] mean_f1={sum(f1s)/len(f1s):.3f} "
          f"mean_rr_mae_ms={(sum(rr)/len(rr)) if rr else float('nan'):.1f} "
          f"n_trials={len(results)} rr_dropped={len(results)-len(rr)}")
    with open("reports/autocorr_baseline_eval.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
