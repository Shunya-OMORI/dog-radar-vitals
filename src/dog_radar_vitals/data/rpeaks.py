"""ECG波形からR波（心拍の基準点）を検出し、RR Intervalを計算する。

RR Interval予測は、密な波形再構成（`ecg_cnn1d.py`）とは異なる定式化
（疎なイベント検出）になりやすいという仮説を検証するため、Schellenbergerの
ヒトECGデータからR波の真値を作る。

検出法は単純な閾値+不応期方式（z-score化したECGでheight>2、最小間隔300ms）。
臨床用ECG（SNRが高い）ではこれで概ね十分だが、Pan-Tompkinsのような正式なQRS検出器
ではない点に注意。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

MIN_RR_SEC = 0.3  # 生理的に妥当な最小RR間隔（=最大200bpm相当）
PEAK_HEIGHT_ZSCORE = 2.0


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """ECG波形（生値）からR波のサンプルインデックスを検出する。"""
    z = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
    peaks, _ = find_peaks(z, height=PEAK_HEIGHT_ZSCORE, distance=int(MIN_RR_SEC * fs))
    return peaks


def rr_intervals_ms(peak_indices: np.ndarray, fs: int) -> np.ndarray:
    """隣接するR波間の時間間隔[ms]を返す（長さ=len(peak_indices)-1）。"""
    return np.diff(peak_indices) / fs * 1000.0


def build_peak_heatmap(peak_indices: np.ndarray, length: int, fs: int, sigma_ms: float = 10.0) -> np.ndarray:
    """R波位置に立てたガウシアンを重ね合わせた1次元heatmap（値域[0,1]）を作る。

    密な波形そのものではなく「R波がいつ起きたか」だけを回帰対象にすることで、
    QRSの振幅・形状を無視してタイミング検出に特化したターゲット表現になる。
    """
    heatmap = np.zeros(length, dtype=np.float32)
    sigma_samples = sigma_ms / 1000.0 * fs
    window = int(sigma_samples * 4)
    t = np.arange(-window, window + 1)
    gaussian = np.exp(-0.5 * (t / sigma_samples) ** 2)

    for idx in peak_indices:
        start = max(0, idx - window)
        end = min(length, idx + window + 1)
        g_start = start - (idx - window)
        g_end = g_start + (end - start)
        heatmap[start:end] = np.maximum(heatmap[start:end], gaussian[g_start:g_end])

    return heatmap


def extract_peaks_from_heatmap(heatmap: np.ndarray, fs: int, height: float = 0.3) -> np.ndarray:
    """予測heatmapからピーク位置を再抽出する（build_peak_heatmapの逆操作に相当）。"""
    peaks, _ = find_peaks(heatmap, height=height, distance=int(MIN_RR_SEC * fs))
    return peaks


def match_peaks(true_peaks: np.ndarray, pred_peaks: np.ndarray, fs: int, tolerance_ms: float = 50.0) -> dict:
    """真のR波と予測R波を、許容誤差内で1対1に対応付ける（貪欲法、時刻順）。

    2つの波形再構成アプローチ（密な波形回帰 vs heatmap回帰）を、検出したR波の
    タイミング精度・RR Intervalの精度という共通の物差しで比較するために使う。
    """
    tolerance_samples = tolerance_ms / 1000.0 * fs
    true_sorted = np.sort(true_peaks)
    pred_sorted = np.sort(pred_peaks)

    matched_true, matched_pred = [], []
    i, j = 0, 0
    while i < len(true_sorted) and j < len(pred_sorted):
        diff = pred_sorted[j] - true_sorted[i]
        if abs(diff) <= tolerance_samples:
            matched_true.append(true_sorted[i])
            matched_pred.append(pred_sorted[j])
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1

    n_true, n_pred, n_matched = len(true_sorted), len(pred_sorted), len(matched_true)
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    timing_errors_ms = None
    if n_matched > 0:
        timing_errors_ms = (np.array(matched_pred) - np.array(matched_true)) / fs * 1000.0

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_true": np.array(matched_true),
        "matched_pred": np.array(matched_pred),
        "timing_errors_ms": timing_errors_ms,
    }


def matched_rr_interval_mae_ms(match_result: dict, fs: int) -> float | None:
    """マッチした真のR波の隣接ペアそれぞれについて、対応する予測RR IntervalとのMAE[ms]を返す。

    真の隣接R波2つが両方ともマッチできていた区間だけを比較対象にする（見逃し・過検出の
    影響を、RR Interval自体の精度評価からできるだけ切り離すため）。
    """
    matched_true = match_result["matched_true"]
    matched_pred = match_result["matched_pred"]
    if len(matched_true) < 2:
        return None

    true_rr = np.diff(matched_true) / fs * 1000.0
    pred_rr = np.diff(matched_pred) / fs * 1000.0
    return float(np.abs(true_rr - pred_rr).mean())
