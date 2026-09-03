"""手作り特徴量+勾配ブースティング(sklearn、深層学習なし)でR波heatmap回帰を
学習・評価する。config284と同一のtrain/val/test分割・同一の評価経路(F1/RR-MAE)。
GPU不要、CPUのみ。
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects
from dog_radar_vitals.data.rpeaks import detect_r_peaks, extract_peaks_from_heatmap, match_peaks, matched_rr_interval_mae_ms
from dog_radar_vitals.models.classical.feature_gbm_rri import build_training_pairs, extract_features

RAW_ROOT = "data/raw"
TRAIN_SUBJECTS = [1, 2, 5, 9, 10, 13, 14]
VAL_SUBJECTS = [16]
TEST_SUBJECTS = [17, 29, 30]


def main() -> None:
    t0 = time.time()
    train_ids = trial_ids_for_subjects(RAW_ROOT, TRAIN_SUBJECTS)
    print(f"train trials: {len(train_ids)}")

    X_list, y_list = [], []
    for tid in train_ids:
        rec = load_trial(RAW_ROOT, tid)
        X, y = build_training_pairs(rec.rcg, rec.ecg, rec.fs)
        X_list.append(X)
        y_list.append(y)
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    print(f"training pairs: {X_train.shape}, positive fraction (>0.1): {(y_train > 0.1).mean():.3f}")

    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"train done in {time.time() - t0:.1f}s")

    joblib.dump(model, "reports/feature_gbm_rpeak_model.joblib")

    # 評価: test被験者の全トライアルで、区切りなしの連続予測(モデル本体と同じ評価プロトコル)
    test_ids = trial_ids_for_subjects(RAW_ROOT, TEST_SUBJECTS)
    results = []
    for tid in test_ids:
        rec = load_trial(RAW_ROOT, tid)
        feats = extract_features(rec.rcg, rec.fs)
        pred_heatmap = np.clip(model.predict(feats), 0.0, 1.0)

        ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
        true_peaks = detect_r_peaks(ecg_filled, rec.fs)
        pred_peaks = extract_peaks_from_heatmap(pred_heatmap, rec.fs, height=0.3)
        match = match_peaks(true_peaks, pred_peaks, rec.fs, tolerance_ms=50)
        rr_mae = matched_rr_interval_mae_ms(match, rec.fs)
        results.append({"trial_id": tid, "f1": match["f1"], "precision": match["precision"],
                         "recall": match["recall"], "rr_mae_ms": rr_mae})
        print(f"trial {tid}: f1={match['f1']:.3f} precision={match['precision']:.3f} "
              f"recall={match['recall']:.3f} rr_mae_ms={rr_mae}")

    f1s = [r["f1"] for r in results]
    rr = [r["rr_mae_ms"] for r in results if r["rr_mae_ms"] is not None]
    print(f"\n[feature+GBM(非深層学習)] mean_f1={sum(f1s)/len(f1s):.3f} "
          f"mean_rr_mae_ms={(sum(rr)/len(rr)) if rr else float('nan'):.1f} "
          f"n_trials={len(results)} rr_dropped={len(results)-len(rr)}")
    print(f"total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
