"""RCG(50ch) -> R波heatmapの窓切り出し。`rpeak_windowing.py`（Schellenberger）と対になる。"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from dog_radar_vitals.data.bandpass import bandpass_filter, ema_clutter_removal, hilbert_envelope
from dog_radar_vitals.data.channel_weighting import apply_channel_weights, compute_channel_weights
from dog_radar_vitals.data.mmecg import MMECGRecording
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, robust_scale_channels, zscore_channels
from dog_radar_vitals.data.rpeaks import (
    build_peak_box,
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
    heatmap_sigma_ms: float = 10.0,
    normalize: str = "zscore",
    target_mode: str = "all_peaks",
    target_shape: str = "gaussian",
    box_half_width_ms: float = 150.0,
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
        心拍帯パワー比が高い点を重視するSNRベースの重み付け。"Cardio-Focusing"型の発想)、
        "hilbert_envelope"(2026-08-28、`bandpass.hilbert_envelope`。心拍帯バンドパス後の
        Hilbert変換包絡線に置き換える。IMU/BCGの心拍推定やレーダバイタルサイン計測で
        包絡線を重視する先行研究がある：Makwana ら, IJECE 2016／Choudhary ら,
        arXiv:1809.03174。生波形の符号や位相ではなく「瞬間的な揺れの強さ」を
        直接入力にする)。
    heatmap_sigma_ms: 2026-08-07、教師heatmapのGaussianラベル幅(既定10ms、
        `build_peak_heatmap`参照)。前処理(channel_weight/ema_clutter/bandpass)は
        いずれも274(前処理なし)を上回れなかったため、入力側ではなくラベル側
        (R5: モデルより先に入力と評価を疑う)の単一変数として追加した。
    target_mode: 2026-08-27、ユーザ指示「教師を一番近いRの点にする」への対応。
        "all_peaks"(既定): 窓内に含まれる全R波にガウシアンを立てる(現行採用)。
        "nearest_peak": 窓中心に最も近いR波1点だけにガウシアンを立て、窓内の
        他のR波は無視する(prepare_mmecg_dataset.py の ecg_seg/PPI 構築で既に
        使っている「窓中心に最も近いピークを基準ビートとする」という考え方を、
        anchor/heatmapタスク側にも適用したもの)。「1窓=1拍を検出する」という
        単純化されたタスク定式化になり、複数拍を一度に検出する現行方式との
        対照実験に使う。
    target_shape: 2026-08-28、教師の"形"自体を差し替える(target_modeとは独立の軸)。
        "gaussian"(既定): build_peak_heatmap、sigma_ms幅のガウシアン(現行採用)。
        "box": build_peak_box、box_half_width_ms幅の矩形0/1マスク。QRS検出の
        U-Net系文献(Sereda ら, arXiv:1912.09223 等)で一般的な教師表現との比較用。
        train.loss="box_bce_dice"と組み合わせて使う想定。
    box_half_width_ms: target_shape="box"のときの矩形の半値幅[ms](既定150ms)。
    """
    nan_mask = None
    if np.isnan(rec.rcg).any() or np.isnan(rec.ecg).any():
        nan_mask = np.isnan(rec.rcg).any(axis=1) | np.isnan(rec.ecg)

    # 各前処理は排他的に適用する(優先順位: channel_weight > ema_clutter > hilbert_envelope > apply_bandpass > none)。
    # 複数同時に有効化した場合は単一変数プローブの前提が崩れるため、意図的に併用しない。
    rcg_for_norm = rec.rcg
    if preprocess in ("channel_weight", "ema_clutter", "hilbert_envelope") or apply_bandpass:
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
        elif preprocess == "hilbert_envelope":
            rcg_for_norm = np.stack(
                [hilbert_envelope(rcg_filled[:, ch], rec.fs) for ch in range(rcg_filled.shape[1])], axis=-1)
        else:
            rcg_for_norm = np.stack(
                [bandpass_filter(rcg_filled[:, ch], rec.fs) for ch in range(rcg_filled.shape[1])], axis=-1)
    # normalize (2026-08-25): 既定は従来どおり z-score。"robust" は中央値と MAD で正規化する。
    # 検出が崩れている被験者(9, 16)では、分散最大チャネルに心拍と無関係な巨大スパイクが
    # 散発しており、標準偏差がそれに膨らんで心拍成分が相対的に潰れることを目視で確認した。
    rcg_z = robust_scale_channels(rcg_for_norm) if normalize == "robust" else zscore_channels(rcg_for_norm)
    rcg_input = analytic_signal_channels(rcg_z) if complex_input else rcg_z

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    if detector == "neurokit_rt":
        r_peaks, t_peaks = detect_r_and_t_peaks_neurokit(ecg_filled, rec.fs)
        peak_indices = np.sort(np.concatenate([r_peaks, t_peaks]))
    else:
        peak_fn = detect_r_peaks_neurokit if detector == "neurokit" else detect_r_peaks
        peak_indices = peak_fn(ecg_filled, rec.fs)
    if target_shape == "box":
        heatmap = build_peak_box(peak_indices, length=len(ecg_filled), fs=rec.fs, half_width_ms=box_half_width_ms)
    else:
        heatmap = build_peak_heatmap(peak_indices, length=len(ecg_filled), fs=rec.fs, sigma_ms=heatmap_sigma_ms)

    window_len = int(window_sec * rec.fs)
    stride = int(stride_sec * rec.fs)

    n_steps = rcg_input.shape[0]
    for start in range(0, n_steps - window_len + 1, stride):
        end = start + window_len
        if nan_mask is not None and nan_mask[start:end].any():
            continue
        if target_mode == "nearest_peak":
            center = start + window_len // 2
            if len(peak_indices) == 0:
                window_heatmap = np.zeros(window_len, dtype=np.float32)
            else:
                nearest = peak_indices[np.argmin(np.abs(peak_indices - center))]
                local_idx = np.array([nearest - start])
                window_heatmap = build_peak_heatmap(local_idx, length=window_len, fs=rec.fs, sigma_ms=heatmap_sigma_ms)
        else:
            window_heatmap = heatmap[start:end]
        yield rcg_input[start:end], window_heatmap
