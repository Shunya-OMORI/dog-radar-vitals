"""RCG(50ch) -> ECG波形の窓切り出し。`ecg_windowing.py`（Schellenberger、2ch I/Q）と対になる。

`complex_input=True`のとき、各チャネルにHilbert変換を適用してanalytic signal（複素数）化した
うえで窓を切り出す（複素領域モデル用）。窓境界でのHilbert変換アーチファクトを避けるため、
変換は窓切り出し前の録音全体に対して1回だけ行う。
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.signal import hilbert

from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap
from dog_radar_vitals.data.mmecg import MMECGRecording


def zscore_channels(rcg: np.ndarray) -> np.ndarray:
    """RCG(T, 50)をチャネルごとにz-score正規化する。`mmecg_rpeak_windowing.py`とも共有する。"""
    return np.stack([zscore_with_nan_gap(rcg[:, ch]) for ch in range(rcg.shape[1])], axis=-1)


def minmax_with_nan_gap(x: np.ndarray) -> np.ndarray:
    """1次元信号を[-1, 1]にmin-max正規化する（NaNは統計量計算からnanmin/nanmaxで除外）。

    ユーザ経験則（IMU/レーダ入力・ECG出力とも0-1または-1-1のmin-max正規化が良い）と、
    Radar2ECG(bottleneck fusion論文)の前処理方針を踏まえた代替正規化。z-scoreと異なり
    外れ値1点でスケールが決まるため、レーダのモーションアーチファクト等の突発的な
    外れ値が録音に含まれると正規化後の実効ダイナミックレンジが縮む点に注意。
    """
    lo, hi = np.nanmin(x), np.nanmax(x)
    return 2.0 * (x - lo) / (hi - lo + 1e-8) - 1.0


def minmax_channels(rcg: np.ndarray) -> np.ndarray:
    """RCG(T, 50)をチャネルごとに[-1, 1]にmin-max正規化する。"""
    return np.stack([minmax_with_nan_gap(rcg[:, ch]) for ch in range(rcg.shape[1])], axis=-1)


def analytic_signal_channels(rcg_z: np.ndarray) -> np.ndarray:
    """z-score済みRCG(T, 50)をチャネルごとにHilbert変換し、complex64の(T, 50)を返す。"""
    return hilbert(rcg_z, axis=0).astype(np.complex64)


def iter_windows(
    rec: MMECGRecording,
    window_sec: float,
    stride_sec: float,
    complex_input: bool = False,
    normalization: str = "zscore",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(RCG窓 [T,50](real or complex64), ECG窓 [T]) を順に返す。

    `normalization`は"zscore"（デフォルト、平均0分散1）または"minmax"（[-1,1]、
    録音全体のmin/maxを基準にRCG各チャネル・ECGをそれぞれ正規化）。
    """
    if normalization not in ("zscore", "minmax"):
        raise ValueError(f"unknown normalization: {normalization}")

    nan_mask = None
    if np.isnan(rec.rcg).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.rcg).any(axis=1) | np.isnan(rec.ecg)

    if normalization == "zscore":
        rcg_norm = zscore_channels(rec.rcg)
        ecg = zscore_with_nan_gap(rec.ecg)
    else:
        rcg_norm = minmax_channels(rec.rcg)
        ecg = minmax_with_nan_gap(rec.ecg)
    rcg_input = analytic_signal_channels(rcg_norm) if complex_input else rcg_norm

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = rcg_input.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        yield rcg_input[start:end], ecg[start:end]
