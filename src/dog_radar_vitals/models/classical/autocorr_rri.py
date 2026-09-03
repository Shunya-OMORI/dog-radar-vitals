"""レーダ心拍計測分野で標準的な、非学習(古典信号処理)のR波/RR間隔推定。

背景 (2026-08-28、ユーザ指示): 「4秒窓を与えているなら、信号処理・数学の分野で
確立された手法があるはず」との指摘への対応。実際、FMCWレーダの心拍/呼吸推定では
自己相関(autocorrelation)またはFFTでの周期推定が標準的な古典手法である
(例: Nonlinear Spectral Approach for Radar-Based Heartbeat Estimation,
arXiv:2507.20664; FACE法, IEEE 2012)。CLAUDE.md R3が要求する「入力を見ずに
平均だけ出すモデル」に相当する、本タスクでの非学習ベースラインとしても使う
(深層学習モデルがこれを上回っているかを必ず確認する)。

手順:
  1. channel_weighting.compute_channel_weights で50点から心拍帯SNRが高い点を選び、
     重み付き平均で1チャンネルに集約する(既存のchannel_weight前処理と同じ発想)。
  2. bandpass_filter で心拍帯([1,25]Hz)を抽出する(既存のapply_bandpassと同じ関数)。
  3. 短時間自己相関(窓幅4秒)で生理的に妥当なRR間隔(300-1500ms)の範囲内の
     最大自己相関ラグを周期候補として推定する。
  4. 3で得た周期を最小ピーク間隔の制約として使い、帯域通過後の信号にfind_peaksを
     適用してR波位置を検出する(周期推定と山検出を分離した、教科書的な2段構成)。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import correlate as scipy_correlate

from dog_radar_vitals.data.bandpass import bandpass_filter
from dog_radar_vitals.data.channel_weighting import apply_channel_weights, compute_channel_weights

MIN_RR_SEC = 0.3
MAX_RR_SEC = 1.5


def estimate_period_autocorr(x: np.ndarray, fs: int, min_rr_sec: float = MIN_RR_SEC, max_rr_sec: float = MAX_RR_SEC) -> float:
    """自己相関から生理的に妥当な範囲内の周期[秒]を1つ推定する。"""
    # 2026-08-28: 直接法(np.correlate, O(n^2))は数万サンプルのトライアルで極端に遅く、
    # GPU学習ジョブのCPUデータローダと競合した。FFTベース(scipy.signal.correlate,
    # method='fft', O(n log n))に差し替え(計算内容は同一)。
    x = x - x.mean()
    n = len(x)
    if n < int(max_rr_sec * fs) * 2:
        return (min_rr_sec + max_rr_sec) / 2.0
    acf = scipy_correlate(x, x, mode="full", method="fft")[n - 1:]
    acf = acf / (acf[0] + 1e-12)
    lag_min, lag_max = int(min_rr_sec * fs), min(int(max_rr_sec * fs), n - 1)
    if lag_max <= lag_min:
        return (min_rr_sec + max_rr_sec) / 2.0
    best_lag = lag_min + int(np.argmax(acf[lag_min:lag_max]))
    return best_lag / fs


def detect_peaks_autocorr(rcg: np.ndarray, fs: int) -> np.ndarray:
    """RCG(T, n_points) から、古典的な自己相関ベースの手法でR波位置を検出する。

    深層学習モデルを一切使わない、CLAUDE.md R3の「非学習ベースライン」に相当する。
    """
    weights = compute_channel_weights(rcg, fs)
    weighted = apply_channel_weights(rcg, weights)  # (T, n_points)
    combined = weighted.sum(axis=1) / (weights.sum() + 1e-12)  # 重み付き平均で1チャンネルに集約
    filtered = bandpass_filter(combined, fs)

    period_sec = estimate_period_autocorr(filtered, fs)
    min_distance = max(1, int(period_sec * 0.6 * fs))  # 推定周期の6割を最小間隔とし、過検出を抑える

    peaks, _ = find_peaks(np.abs(filtered), distance=min_distance,
                           height=np.percentile(np.abs(filtered), 60))
    return peaks
