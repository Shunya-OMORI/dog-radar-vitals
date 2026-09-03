"""2026-08-28: 今日の312/313/317/318/319比較が、採用中パイプラインに含まれる
順位正規化(normalize_heatmap, rank)と重心復号(refine_peaks_centroid)を欠いた
簡易評価(evaluate_trial_peak_detection)で行われていたと判明(ユーザ指摘)。
「現行」として出した19.1msは簡易評価版の数字で、実際に採用・報告しているのは
LOSOでRR-MAE 10.50ms(50ms許容)〜14.56ms(150ms許容)。

ここでは312/313/317/318/319の各チェックポイントに、採用中と同じ後処理
(rank正規化 -> しきい値 -> 重心復号)を適用し直し、公平な「フル後処理」条件
(単一split、valでしきい値選択)で再評価する。
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

import numpy as np
import torch

from dog_radar_vitals.config import load_config
from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, zscore_channels
from dog_radar_vitals.data.rpeaks import (
    detect_r_peaks, extract_peaks_from_heatmap, match_peaks,
    matched_rr_interval_mae_ms, normalize_heatmap, refine_peaks_centroid,
)
from dog_radar_vitals.mmecg_rpeak_evaluation import _predict_full_recording, load_mmecg_model

RUNS = {
    "312_full_teacherfix": ("runs/20260828-115510_ecg_rpeak_cnn1d_312_mmecg_rpeak_cnn1d_config284match_64ch_full_teacherfix", "configs/experiments/312_mmecg_rpeak_cnn1d_config284match_64ch.yaml"),
    "313_full_teacherfix": ("runs/20260828-115654_ecg_rpeak_cnn1d_313_mmecg_rpeak_cnn1d_config284match_16ch_full_teacherfix", "configs/experiments/313_mmecg_rpeak_cnn1d_config284match_16ch.yaml"),
    "317_unet_full": ("runs/20260828-115813_ecg_rpeak_unet1d_317_mmecg_rpeak_unet1d_config284match_full_teacherfix", "configs/experiments/317_mmecg_rpeak_unet1d_config284match.yaml"),
    "318_attn_unet_full": ("runs/20260828-120032_ecg_rpeak_attention_unet1d_318_mmecg_rpeak_attention_unet1d_config284match_full_teacherfix", "configs/experiments/318_mmecg_rpeak_attention_unet1d_config284match.yaml"),
    "284_baseline": ("runs/20260807-062112_ecg_rpeak_spatial_gnn_284_sigma15_full", "runs/20260807-062112_ecg_rpeak_spatial_gnn_284_sigma15_full/config.yaml"),
}


def eval_one(run_dir: str, config_path: str, height: float, subjects: list[int]) -> list[dict]:
    config = load_config(config_path)
    model = load_mmecg_model(__import__("pathlib").Path(run_dir), config)
    data_cfg = config["data"]
    use_posxyz = config["model"]["family"] in ("mmecg_spatial_seq2seq", "mmecg_spatial_heatmap_gan")
    window_sec = data_cfg["window_sec"]
    trial_ids = trial_ids_for_subjects("data/raw", subjects)

    results = []
    for tid in trial_ids:
        rec = load_trial("data/raw", tid)
        rcg_z = zscore_channels(rec.rcg)
        window_len = int(window_sec * rec.fs)
        pred = _predict_full_recording(model, rcg_z, window_len, posxyz=rec.posxyz if use_posxyz else None)
        pred_norm = normalize_heatmap(pred, rec.fs, kind="rank")

        ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
        true_peaks = detect_r_peaks(ecg_filled, rec.fs)
        coarse = extract_peaks_from_heatmap(pred_norm, rec.fs, height=height)
        refined = refine_peaks_centroid(pred_norm, coarse, rec.fs)
        refined = np.round(refined).astype(int)

        match = match_peaks(true_peaks, refined, rec.fs, tolerance_ms=50)
        rr_mae = matched_rr_interval_mae_ms(match, rec.fs)
        results.append({"trial_id": tid, "f1": match["f1"], "rr_mae_ms": rr_mae})
    return results


def mean_f1(results):
    return sum(r["f1"] for r in results) / len(results)


def mean_rr(results):
    rr = [r["rr_mae_ms"] for r in results if r["rr_mae_ms"] is not None]
    return (sum(rr) / len(rr) if rr else None), len(results) - len(rr)


def main() -> None:
    for name, (run_dir, config_path) in RUNS.items():
        val_scores = {}
        for h in [0.5, 0.7, 0.9, 0.95]:
            val_results = eval_one(run_dir, config_path, h, [16])
            val_scores[h] = mean_f1(val_results)
        best_h = max(val_scores, key=val_scores.get)
        test_results = eval_one(run_dir, config_path, best_h, [17, 29, 30])
        f1 = mean_f1(test_results)
        rr, dropped = mean_rr(test_results)
        print(f"{name}: best_h(val)={best_h} test_f1={f1:.3f} test_rr_mae_ms={rr:.2f} rr_dropped={dropped}/{len(test_results)}")


if __name__ == "__main__":
    main()
