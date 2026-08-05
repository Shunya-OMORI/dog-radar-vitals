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
from scipy.signal import butter, sosfiltfilt

BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 25.0
BANDPASS_ORDER = 4


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
