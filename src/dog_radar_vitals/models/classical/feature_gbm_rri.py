"""手作り特徴量 + 勾配ブースティング(sklearn, 深層学習なし)によるR波heatmap回帰。

背景 (2026-08-28、ユーザ指示・指導教員の指摘への対応): 「機械学習だとしても
深層学習に拘る必要はない」。先行研究にも学習ベースだが深層学習ではないものがある
可能性を踏まえ、古典的な特徴量工学+浅いモデル(sklearn HistGradientBoostingRegressor、
勾配ブースティング木)で同じタスク(R波heatmap回帰)を解けるかを検証する。

深層学習モデル(rpeak_spatial_gnn等)との違い:
    - 特徴量は手作り(生波形を直接読ませない)。心拍帯のエネルギー・包絡線・
      局所勾配など、QRS検出の古典文献で使われる特徴量に相当するものを使う
    - モデルは決定木のアンサンブル(勾配ブースティング)。畳み込み・注意機構は
      一切使わない。パラメータではなく決定木の分岐条件で表現力を持つ
    - 学習後は木の集合として保存でき、GPU不要でCPU上で高速に推論できる
      (エッジ実行という研究目的そのものと相性が良い可能性がある)

対応する先行研究: 勾配ブースティング木(Friedman, "Greedy Function Approximation:
A Gradient Boosting Machine," Ann. Statist., 2001)によるheatmap回帰は、
姿勢推定分野でもCNN以前に使われていた(Random Forest回帰によるkeypoint検出、
Criminisi ら, "Decision Forests," Microsoft Research, 2011 系譜)。
"""
from __future__ import annotations

import numpy as np

from dog_radar_vitals.data.bandpass import bandpass_filter, hilbert_envelope
from dog_radar_vitals.data.channel_weighting import apply_channel_weights, compute_channel_weights
from dog_radar_vitals.data.mmecg_windowing import zscore_channels
from dog_radar_vitals.data.rpeaks import build_peak_heatmap, detect_r_peaks_neurokit


def _rolling_rms(x: np.ndarray, half_win: int) -> np.ndarray:
    """局所RMSエネルギー(Shannon energy envelopeに相当する古典的QRS特徴量)。

    scipy.ndimage.uniform_filter1dで移動平均をベクトル化(端は反射境界)。
    """
    from scipy.ndimage import uniform_filter1d

    mean_sq = uniform_filter1d(x.astype(np.float64) ** 2, size=2 * half_win + 1, mode="nearest")
    return np.sqrt(mean_sq)


def extract_features(rcg: np.ndarray, fs: int) -> np.ndarray:
    """RCG(T, n_points) -> 特徴量行列(T, n_features)。

    列: [z-score済み最良チャネル値, Hilbert包絡線, 局所RMSエネルギー(±25ms),
         1階差分(局所勾配), 2階差分(局所曲率), 心拍帯パワー上位3チャネルのz-score値]
    """
    weights = compute_channel_weights(rcg, fs)
    weighted = apply_channel_weights(rcg, weights)
    combined = weighted.sum(axis=1) / (weights.sum() + 1e-12)

    filtered = bandpass_filter(combined, fs)
    envelope = hilbert_envelope(combined, fs)
    half_win = max(1, int(0.025 * fs))  # ±25ms
    energy = _rolling_rms(filtered, half_win)

    d1 = np.gradient(filtered)
    d2 = np.gradient(d1)

    top3_idx = np.argsort(weights)[-3:]
    rcg_z = zscore_channels(rcg)
    top3 = rcg_z[:, top3_idx]  # (T, 3)

    feats = np.column_stack([
        zscore_channels(combined[:, None])[:, 0],
        (envelope - envelope.mean()) / (envelope.std() + 1e-8),
        (energy - energy.mean()) / (energy.std() + 1e-8),
        d1 / (d1.std() + 1e-8),
        d2 / (d2.std() + 1e-8),
        top3,
    ])
    return feats.astype(np.float32)


def build_training_pairs(
    rcg: np.ndarray, ecg: np.ndarray, fs: int, heatmap_sigma_ms: float = 15.0,
    subsample_neg_ratio: float = 5.0, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1トライアル分の特徴量・教師を作り、負例(heatmap値がほぼ0の点)を間引く。

    subsample_neg_ratio: 正例(heatmap>0.1)1点あたり、負例を何点残すか。
    R波(σ=15msガウシアン)は4秒窓の1割未満しか正例にならないため、間引かないと
    決定木が「全部0」で損失を最小化してしまう(§3.7と同じクラス不均衡問題)。
    """
    rng = rng or np.random.default_rng(42)
    ecg_filled = np.nan_to_num(ecg, nan=float(np.nanmean(ecg)))
    peaks = detect_r_peaks_neurokit(ecg_filled, fs)
    heatmap = build_peak_heatmap(peaks, length=len(ecg_filled), fs=fs, sigma_ms=heatmap_sigma_ms)

    feats = extract_features(rcg, fs)
    n = min(len(feats), len(heatmap))
    feats, heatmap = feats[:n], heatmap[:n]

    pos_idx = np.where(heatmap > 0.1)[0]
    neg_idx = np.where(heatmap <= 0.1)[0]
    n_neg = min(len(neg_idx), int(len(pos_idx) * subsample_neg_ratio))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else neg_idx
    keep = np.concatenate([pos_idx, neg_sample])
    return feats[keep], heatmap[keep]
