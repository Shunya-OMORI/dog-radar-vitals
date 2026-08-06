"""RCG(50点)のうち心拍帯パワーが強い点を重視する、SNRベースのチャンネル重み付け。

背景（2026-08-07）: RCGの50点は胸郭上のばらばらの位置から返ってきた反射点で、心臓に近い
点ほど心拍由来の変位が強く、遠い点はノイズ寄りと推測される（原著radarODEの3次元位置
埋め込み＋Transformerも同様の直感に基づく設計）。空間GNNはGATConvで空間的な重み付けを
学習できる設計だが、被験者7名という小規模データでは、明示的にSNRの高い点を強調する
前処理が学習を助ける可能性がある。

対応する先行研究: FMCWレーダのバイタルサイン計測分野で標準的な「チャンネル/アンテナ
選択」手法（心拍帯パワー比によるSNR判定、RXチャンネル融合時の重み付け）。特に
"Cardio-Focusing Algorithm"（心拍情報が強い空間位置を動的に特定するアルゴリズム、
レーダ→ECG変換タスク向け）と同種の発想。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

CARDIAC_BAND_LOW_HZ = 0.8
CARDIAC_BAND_HIGH_HZ = 3.0


def cardiac_band_power_ratio(x: np.ndarray, fs: int) -> float:
    """1チャンネルの信号について、心拍帯([0.8,3]Hz)パワー / 全帯域パワーの比を返す。

    値が高いほど、その点は心拍由来の変位を強く含んでいる（＝SNRが高い）と解釈する。
    """
    freqs, psd = welch(x, fs=fs, nperseg=min(len(x), fs * 4))
    total_power = psd.sum()
    if total_power < 1e-12:
        return 0.0
    band_mask = (freqs >= CARDIAC_BAND_LOW_HZ) & (freqs <= CARDIAC_BAND_HIGH_HZ)
    return float(psd[band_mask].sum() / total_power)


def compute_channel_weights(rcg: np.ndarray, fs: int) -> np.ndarray:
    """RCG(T, n_points)の各点について心拍帯パワー比を計算し、[0,1]に正規化した
    重み(n_points,)を返す（最大の点が1.0になるよう正規化）。
    """
    n_points = rcg.shape[1]
    scores = np.array([cardiac_band_power_ratio(rcg[:, ch], fs) for ch in range(n_points)])
    max_score = scores.max()
    if max_score < 1e-12:
        return np.ones(n_points, dtype=np.float32)
    return (scores / max_score).astype(np.float32)


def apply_channel_weights(rcg: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """RCG(T, n_points)の各チャンネルに重み(n_points,)を掛ける(SNRの低い点を減衰)。"""
    return rcg * weights[np.newaxis, :]
