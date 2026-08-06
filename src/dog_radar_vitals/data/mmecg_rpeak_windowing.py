"""RCG(50ch) -> R波heatmapの窓切り出し。`rpeak_windowing.py`（Schellenberger）と対になる。"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dog_radar_vitals.data.bandpass import bandpass_filter, ema_clutter_removal
from dog_radar_vitals.data.channel_weighting import apply_channel_weights, compute_channel_weights
from dog_radar_vitals.data.mmecg import MMECGRecording
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, zscore_channels
from dog_radar_vitals.data.rpeaks import (
    build_peak_heatmap,
    detect_r_and_t_peaks_neurokit,
    detect_r_peaks,
    detect_r_peaks_neurokit,
)


def iter_peak_windows(
    rec: MMECGRecording,
    window_sec: float,
    stride_sec: float,
    complex_input: bool = False,
    detector: str = "legacy",
    apply_bandpass: bool = False,
    preprocess: str = "none",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(RCG窓 [T,50](real or complex64), R波heatmap窓 [T]) を順に返す。

    detector: "legacy"(z-score+find_peaks、既定・過去との再現性のため)、
        "neurokit"(2026-08-06の恒久対応、R/T混同を回避)、
        "neurokit_rt"(2026-08-06のユーザ発案の派生案: R波だけでなくT波も教師
        heatmapに含める。モデルに「T波を無視しろ」という一貫しない教師を
        与え続けるのをやめ、R/Tどちらを検出したかの選別は後処理に任せる。
        評価時の正解はdetect_r_peaks_neurokit(R波のみ)を使うため、この
        モードの効果は学習後の後処理付き評価で確認する)。
    apply_bandpass: 2026-08-07、z-score正規化の前にRCGへ[1,25]Hzバンドパスフィルタ
        (`bandpass.py`)をかけるかどうか。SST/波形回帰パイプラインでは既に試されて
        悪化した記録がある(config253/254)が、別パイプライン・旧ラベルでの結果で
        あり、本パイプライン(生信号heatmap, 新ラベル)での単一変数プローブとして
        別途検証する。既定False(過去との再現性のため)。
    preprocess: "none"(既定)、"ema_clutter"(2026-08-07、`bandpass.ema_clutter_removal`。
        指数移動平均基線を引く適応的クラッタ除去。FMCWレーダのバイタルサイン計測分野で
        使われる、固定次数Butterworthとは異なる高域通過特性を持つ手法。apply_bandpassとは
        併用しない想定)、"channel_weight"(2026-08-07、`channel_weighting.py`。50点のうち
        心拍帯パワー比が高い点を重視するSNRベースの重み付け。"Cardio-Focusing"型の発想)。
    """
    nan_mask = None
    if np.isnan(rec.rcg).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.rcg).any(axis=1) | np.isnan(rec.ecg)

    # 各前処理は排他的に適用する(優先順位: channel_weight > ema_clutter > apply_bandpass > none)。
    # 複数同時に有効化した場合は単一変数プローブの前提が崩れるため、意図的に併用しない。
    rcg_for_norm = rec.rcg
    if preprocess in ("channel_weight", "ema_clutter") or apply_bandpass:
        # sosfiltfilt/ema/welchはNaNが1つでもあると出力全体を汚染しうるため、フィルタ前だけ
        # チャネル平均で一時的に埋める（該当窓はnan_maskで別途スキップされるので
        # ここでの埋め方自体は結果に影響しない）。
        rcg_filled = np.where(np.isnan(rec.rcg), np.nanmean(rec.rcg, axis=0, keepdims=True), rec.rcg)
        if preprocess == "channel_weight":
            weights = compute_channel_weights(rcg_filled, rec.fs)
            rcg_for_norm = apply_channel_weights(rcg_filled, weights)
        elif preprocess == "ema_clutter":
            rcg_for_norm = np.stack(
                [ema_clutter_removal(rcg_filled[:, ch]) for ch in range(rcg_filled.shape[1])], axis=-1)
        else:
            rcg_for_norm = np.stack(
                [bandpass_filter(rcg_filled[:, ch], rec.fs) for ch in range(rcg_filled.shape[1])], axis=-1)
    rcg_z = zscore_channels(rcg_for_norm)
    rcg_input = analytic_signal_channels(rcg_z) if complex_input else rcg_z

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    if detector == "neurokit_rt":
        r_peaks, t_peaks = detect_r_and_t_peaks_neurokit(ecg_filled, rec.fs)
        peak_indices = np.sort(np.concatenate([r_peaks, t_peaks]))
    else:
        peak_fn = detect_r_peaks_neurokit if detector == "neurokit" else detect_r_peaks
        peak_indices = peak_fn(ecg_filled, rec.fs)
    heatmap = build_peak_heatmap(peak_indices, length=len(ecg_filled), fs=rec.fs)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = rcg_input.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        yield rcg_input[start:end], heatmap[start:end]
