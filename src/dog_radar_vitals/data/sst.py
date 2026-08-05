"""Synchrosqueezed Wavelet Transform (SST)による周波数領域への変換。

radarODE(Zhang et al. 2024/2025, arXiv:2408.01672)が採用する前処理で、生の時系列RCGを
そのままモデルに入力する代わりに、SSTでエネルギーを瞬時周波数近傍へ再集中させた
時間周波数スペクトログラムへ変換してから入力する。論文はpower spectrogram entropy(PSE)で
STFT=0.94・CWT=0.90・SST=0.76と比較し、SSTが最もエネルギーが集中した(=クリーンな)表現を
与えることを報告している（詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
「データ規模の再検証」節）。

心拍の心臓力学的振動は[1, 25]Hzに収まるため（論文同様）、この帯域のみを切り出して使う。
"""
from __future__ import annotations

import numpy as np
import ssqueezepy as ssq

FREQ_MIN_HZ = 1.0
FREQ_MAX_HZ = 25.0


def sst_magnitude(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """1chの時系列(T,)をSST変換し、[1, 25]Hzに絞った振幅スペクトログラムを返す。

    戻り値: (magnitude(F, T) float32, freqs(F,) float32 昇順)。
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    tx, _, ssq_freqs, _ = ssq.ssq_cwt(x, wavelet="morlet", fs=fs)
    ssq_freqs = np.asarray(ssq_freqs, dtype=np.float64)

    # ssq_cwt returns frequencies in descending order (high->low); flip to ascending.
    order = np.argsort(ssq_freqs)
    ssq_freqs = ssq_freqs[order]
    tx = tx[order]

    band = (ssq_freqs >= FREQ_MIN_HZ) & (ssq_freqs <= FREQ_MAX_HZ)
    magnitude = np.abs(tx[band]).astype(np.float32)
    return magnitude, ssq_freqs[band].astype(np.float32)
