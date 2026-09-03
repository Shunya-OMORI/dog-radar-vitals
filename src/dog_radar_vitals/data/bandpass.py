"""RCGの心拍帯[1,25]Hzバンドパスフィルタ。

2026-07-29の検証で、RCG正規化(z-score)の分散計算が生の広帯域信号(体動由来と推測される
25Hz超の高周波ノイズを含む)に支配されていることが判明した(trial 1 ch0で、10秒窓ごとの
標準偏差が生広帯域では最大9.3倍ばらつくのに対し、バンドパス後は3.4倍まで縮小)。
呼吸帯(0.1-0.5Hz)・DC帯(0-0.1Hz)のパワーは元々無視できるレベルだったため、
このフィルタは主に高周波ノイズの除去を目的とする(詳細は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 25.0
BANDPASS_ORDER = 4


def hilbert_envelope(x: np.ndarray, fs: int) -> np.ndarray:
    """心拍帯バンドパス後の信号にHilbert変換をかけ、包絡線(解析信号の絶対値)を返す。

    背景 (2026-08-28、ユーザ指摘): IMU→ECG推定やBCG(ballistocardiogram)の心拍推定では
    包絡線を重視する研究例があり、レーダのバイタルサイン計測でもHilbert変換による
    包絡線抽出が使われている。R波(QRS)は信号のエネルギーが瞬間的に高まる区間なので、
    生波形の符号や位相ではなく「その瞬間どれだけ揺れが強いか」という包絡線のほうが、
    ピーク位置の手がかりとして直接的である可能性がある。

    先にbandpass_filter([1,25]Hz)で心拍帯に絞ってから包絡線を取る。生の広帯域信号は
    エネルギーの98%が3-25Hzの高域雑音にあることが実測で分かっており(§5.3(d)の
    RCGスペクトル解析)、フィルタなしで包絡線を取るとその雑音のエンベロープを
    拾ってしまう。

    対応する先行研究: Makwana ら, "Hilbert Transform Based Adaptive ECG R-Peak
    Detection Technique," IJECE, 2016（Hilbert変換の包絡線でR波を直接検出する
    古典手法）／Choudhary ら, "Heart Rate Estimation from Ballistocardiography
    Based on Hilbert Transform and Phase Vocoder," arXiv:1809.03174, 2018
    （BCGでHilbert包絡線から心拍を推定）。
    """
    filtered = bandpass_filter(x, fs)
    return np.abs(hilbert(filtered))


def bandpass_filter(x: np.ndarray, fs: int) -> np.ndarray:
    """1次元信号(T,)に[1,25]Hzのゼロ位相バンドパスフィルタをかける。

    短い区間(単一拍セグメント等)でも安全に動くよう、padlenを信号長に応じて調整する。
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    sos = butter(BANDPASS_ORDER, [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), max(len(x) - 1, 0))
    if len(x) <= padlen or len(x) < 8:
        return x.copy()
    return sosfiltfilt(sos, x, padlen=padlen)


EMA_CLUTTER_ALPHA = 0.01


def ema_clutter_removal(x: np.ndarray, alpha: float = EMA_CLUTTER_ALPHA) -> np.ndarray:
    """指数移動平均(EMA)基線を引く、適応的なクラッタ(静止反射)除去（2026-08-07）。

    背景: FMCWレーダのバイタルサイン計測分野で、静止反射物由来のDC/低周波オフセットを
    抑える定番の手法（固定次数のButterworthバンドパスとは異なり、基線が緩やかに追従する
    一次遅れ系）。Butterworthバンドパス（config253/254、278で悪化を確認済み）とは
    フィルタ特性が異なるため、別の単一変数プローブとして試す価値がある。

    対応する先行研究: FMCWレーダ生体信号処理におけるEMAベースの静止クラッタ抑制
    （高域通過特性を持つ一次IIRフィルタとしての利用、非同期移動平均法）。

    実装: y[t] = x[t] - baseline[t]、baseline[t] = alpha*x[t] + (1-alpha)*baseline[t-1]。
    alphaが小さいほど基線がゆっくり追従＝カットオフ周波数が低い高域通過フィルタに相当する。
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    baseline = np.empty_like(x)
    baseline[0] = x[0]
    for t in range(1, len(x)):
        baseline[t] = alpha * x[t] + (1 - alpha) * baseline[t - 1]
    return x - baseline
