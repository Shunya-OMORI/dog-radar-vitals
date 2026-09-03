"""包絡線 + 動的計画法(Viterbi型)による、深層学習を一切使わないR波検出。

背景 (2026-08-28、ユーザ指示・指導教員からの指摘への対応):
    「深層学習にこだわる理由がない」「軽量性を意識した先行研究には深層学習すら
    使っていないものがある(HEBR・mmCG等、いずれも学習なしの信号処理)」との指摘を
    受け、それら先行研究のアプローチを実際に手元で実装する。

    HEBR(Li ら, IoT-J 2026)の抄録記述「動的計画法で高調波支持集合を追跡し、
    位相整合帯域制限再構成でビート波形を得る」を、本コードベースの既存部品で
    再構成した。DTM_info等HEBR独自の内部処理は非公開のため完全な再現ではないが、
    「周期性の手がかり(ここではHilbert包絡線)」+「動的計画法による系列制約」
    という設計思想は共通する。

    以前(072)の単純な自己相関ベースライン(`autocorr_rri.py`, F1=0.090)は、
    周期推定した1つの値を使って単純peak-pickingするだけだった。ここでは
    (a) 心拍帯バンドパス後のHilbert包絡線(`bandpass.hilbert_envelope`、
        2026-08-28実装、深層学習モデル入力として試して悪化した=319)を
        「信頼度マップ」として使い、
    (b) 録画ごとのrank正規化(`normalize_heatmap`、既存)で振幅のばらつきを吸収し、
    (c) 深層学習モデルの出力heatmapを復号するのに使っている既存のVitebi型DP復号器
        (`extract_peaks_viterbi`、Viterbi 1967 / Coast ら 1990)を、
        **深層学習モデルの出力の代わりに包絡線に対してそのまま適用**する。
    モデルの学習は一切行わない。パラメータ数はゼロ。
"""
from __future__ import annotations

import numpy as np

from dog_radar_vitals.data.bandpass import hilbert_envelope
from dog_radar_vitals.data.channel_weighting import apply_channel_weights, compute_channel_weights
from dog_radar_vitals.data.rpeaks import extract_peaks_viterbi, normalize_heatmap


def detect_peaks_envelope_viterbi(
    rcg: np.ndarray,
    fs: int,
    prior_sigma_sec: float = 0.25,
    prior_weight: float = 1.0,
) -> np.ndarray:
    """RCG(T, n_points) から、包絡線+Viterbi型DPでR波位置を検出する(非学習)。"""
    weights = compute_channel_weights(rcg, fs)
    weighted = apply_channel_weights(rcg, weights)
    combined = weighted.sum(axis=1) / (weights.sum() + 1e-12)  # 重み付き平均で1チャンネルに集約

    envelope = hilbert_envelope(combined, fs)
    envelope_norm = normalize_heatmap(envelope, fs, kind="rank")  # [0,1]、録画ごとのrank正規化

    return extract_peaks_viterbi(
        envelope_norm, fs,
        candidate_height=0.5,  # rank正規化後は0.5=中央値。中央値より上の山だけを候補にする
        prior_sigma_sec=prior_sigma_sec, prior_weight=prior_weight,
    )
