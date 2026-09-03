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


def robust_scale_channels(rcg: np.ndarray, eps: float = 1e-8, clip: float = 10.0) -> np.ndarray:
    """RCG(T, 50)を、チャネルごとに中央値と中央絶対偏差(MAD)で正規化する (2026-08-25 追加)。

    背景（目視診断から）: 検出が崩れている被験者(9, 16)のレーダ波形を描いてみると、
    分散が最大のチャネルに **心拍と無関係な巨大スパイク** が散発していた。
    被験者16では、ほぼ全ての正解R位置にモデル出力の小さなピークが立っている
    (適合率0.940 = しきい値を超えた分は正しい)のに、高さが 0.05〜0.2 しかなく
    しきい値0.3を超えられない。超えているのは巨大スパイクがあった拍だけだった。

    `zscore_channels` はチャネルごとの標準偏差で割る。標準偏差は外れ値の二乗で効くので、
    **散発する巨大スパイクが1つあるだけでそのチャネルのスケールが膨らみ、
    割ったあとの心拍成分が相対的に潰される。**
    中央絶対偏差は順序統計量なので、外れ値の大きさに引きずられない
    (breakdown point 50%)。正規分布なら MAD x 1.4826 が標準偏差に一致するので、
    その係数を掛けて z-score とスケールを揃える。

    対応する先行研究: Huber, "Robust Statistics," Wiley, 1981（頑健統計の標準）。
    レーダのバイタルサイン計測でも、静止反射体や体動由来の外れ値を除いてから
    正規化するのは定石で、Wang ら (Sensors 25(17):5607, 2025) は MTI フィルタで
    静止クラッタを落としてから位相を取り出している。
    """
    out = np.empty_like(rcg, dtype=float)
    for ch in range(rcg.shape[1]):
        x = rcg[:, ch].astype(float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < eps:
            # MAD が潰れる(ほぼ定数のチャネル)場合だけ標準偏差に落とす
            scale = np.nanstd(x)
            if not np.isfinite(scale) or scale < eps:
                scale = 1.0
        out[:, ch] = (x - med) / scale
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None and clip > 0:
        # MAD 正規化はスパイクを「潰さずに残す」ので、そのままだと被験者16の実データで
        # 最大 487 に達し勾配を壊す。心拍成分は |x| の中央値で 0.68 程度なので、
        # 生理的にありえない振幅はここで切る。切る操作自体が頑健化の一部なので、
        # MAD 正規化とクリップはセットで 1 つの前処理として扱う。
        out = np.clip(out, -clip, clip)
    return out


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
