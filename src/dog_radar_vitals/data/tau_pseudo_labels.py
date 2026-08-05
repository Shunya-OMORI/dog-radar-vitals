"""τ(時間遅延)疑似ラベルの複数の生成方式(2026-07-31)。

これまで(config 260-263)は、独立した波形回帰モデル(257)の予測に対してoracle探索で
最良シフトを求める`oracle_best_shift`(`temporal_alignment.py`)のみを疑似ラベルとして
使っていた。しかしユーザ指摘の通り、これは「257というモデルが持つ癖」にτの定義自体が
依存してしまう(257が系統的に早め/遅めに予測する部分もτに混入する)という問題がある。

本モジュールは、257のようなモデルの予測に一切依存しない、ECGの生波形とレーダ由来の
拍境界だけから決まる model-independent なτ疑似ラベルを追加する。

## `rpeak_position_tau`: 拍境界内でのR波位置

`mmecg_singlecycle_dataset._compute_bounds`は、レーダ(SST)のみから`detect_consensus_beat_times`
で拍タイミング(50chの候補ピーク時刻のコンセンサス)を検出し、隣接する拍タイミングの中点を
拍境界として使う。この拍境界がもし常に「ECGのR波ちょうど」にアンカーされていれば、
切り出した固定長ECGセグメント内でのR波位置は常に一定(例えば中央)になるはずである。

実際には(a)レーダのコンセンサスピーク自体が心電図のR波と系統的な時間差(機械的活動が電気的
活動よりτだけ遅れるという生理学的知見、radarODE論文が言及するもの)を持つ可能性、
(b)コンセンサス検出のノイズ、の両方が混ざってセグメント内R波位置がばらつく。このばらつきの
うち(a)の成分こそがradarODEの言うτそのものであり、(b)はノイズ源になる。

この方式は257を含むいかなる学習済みモデルの予測にも依存せず、ECGの生波形(生理的事実)と
レーダのみから決めた拍境界(推論時にも使える情報)だけから計算できるため、257ベースの
oracle shiftよりも「τの定義」として原理的に妥当だと考えられる。
"""
from __future__ import annotations

import numpy as np
import torch

from dog_radar_vitals.data.mmecg_singlecycle_dataset import MMECGSingleCycleDataset
from dog_radar_vitals.data.rpeaks import detect_r_peaks

ECG_FS = 200  # mmecg.FS_HZと同じ


def compute_rpeak_position_tau_labels(
    dataset: MMECGSingleCycleDataset, search_margin_frac: float = 0.5
) -> torch.Tensor:
    """拍境界内でのR波位置から、model-independentなτ疑似ラベル(N,)を計算する。

    各拍について、トライアル全体のECGから検出したR波のうち、拍境界の中心に最も近いものを
    採用し、`(r_idx - ecg_start) / (ecg_end - ecg_start) - 0.5`を返す(0=境界の中心、
    正=R波が境界の後半寄り=境界がR波より早い側にずれている)。境界の外(前後
    `search_margin_frac`倍の余白まで)にR波が見つからない拍は、その拍の中心を代わりに
    使う(=ラベル0、無情報だが学習を壊さないための穏当なフォールバック)。
    """
    # トライアルごとにR波を1回だけ検出してキャッシュする(境界をまたいで独立に検出すると
    # 境界付近のR波を取りこぼす/二重検出しうるため、トライアル全体で検出してから割り当てる)。
    rpeaks_by_trial: dict[int, np.ndarray] = {}
    labels = torch.zeros(len(dataset._index), dtype=torch.float32)
    for i, (trial_id, bounds) in enumerate(dataset._index):
        if trial_id not in rpeaks_by_trial:
            ecg_full = dataset._ecg_by_trial[trial_id]
            rpeaks_by_trial[trial_id] = detect_r_peaks(np.nan_to_num(ecg_full), ECG_FS)
        rpeaks = rpeaks_by_trial[trial_id]

        _, _, ecg_start, ecg_end = bounds
        win_len = ecg_end - ecg_start
        margin = int(win_len * search_margin_frac)
        lo, hi = ecg_start - margin, ecg_end + margin
        candidates = rpeaks[(rpeaks >= lo) & (rpeaks < hi)]
        if len(candidates) == 0:
            continue  # フォールバック: ラベル0のまま

        center = (ecg_start + ecg_end) / 2.0
        r_idx = candidates[np.argmin(np.abs(candidates - center))]
        labels[i] = float((r_idx - ecg_start) / win_len - 0.5)

    return labels


def compute_rpeak_heatmap_labels(
    dataset: MMECGSingleCycleDataset, out_len: int = 200, sigma_samples: float = 8.0
) -> torch.Tensor:
    """拍境界内で検出したR波位置に、リサンプル後の出力軸(長さ`out_len`)上でガウシアンを
    立てたheatmap(N, out_len)を返す。radarODE公式実装(`ZYY0844/radarODE-MTL`)の
    "Anchor"補助タスク(spectrum_dataset.pyのコメント:"anchor_data represent the position
    of the R peak in the ecg signal")と同じ発想: 波形の振幅・形状ではなく「R波がいつ起きたか」
    だけを教師信号にする。値域は`rpeaks.build_peak_heatmap`と同じ[0,1](中心で1、ガウシアンで
    減衰)。R波が見つからない拍はheatmapが全てゼロになる(`rpeak_position_tau`の
    フォールバックと同じ扱い)。
    """
    rpeaks_by_trial: dict[int, np.ndarray] = {}
    heatmaps = torch.zeros(len(dataset._index), out_len, dtype=torch.float32)
    window = int(sigma_samples * 4)
    t = np.arange(-window, window + 1)
    gaussian = np.exp(-0.5 * (t / sigma_samples) ** 2)

    for i, (trial_id, bounds) in enumerate(dataset._index):
        if trial_id not in rpeaks_by_trial:
            ecg_full = dataset._ecg_by_trial[trial_id]
            rpeaks_by_trial[trial_id] = detect_r_peaks(np.nan_to_num(ecg_full), ECG_FS)
        rpeaks = rpeaks_by_trial[trial_id]

        _, _, ecg_start, ecg_end = bounds
        win_len = ecg_end - ecg_start
        margin = int(win_len * 0.5)
        lo, hi = ecg_start - margin, ecg_end + margin
        candidates = rpeaks[(rpeaks >= lo) & (rpeaks < hi)]
        if len(candidates) == 0:
            continue  # フォールバック: heatmap全ゼロのまま

        center = (ecg_start + ecg_end) / 2.0
        r_idx = candidates[np.argmin(np.abs(candidates - center))]
        # 元のサンプル位置をリサンプル後の出力軸(0..out_len-1)に線形変換する。
        resampled_idx = int(round((r_idx - ecg_start) / win_len * out_len))
        if resampled_idx < -window or resampled_idx >= out_len + window:
            continue  # マージン内だが出力軸から大きく外れる場合はフォールバック(heatmap全ゼロ)

        start = max(0, resampled_idx - window)
        end = min(out_len, resampled_idx + window + 1)
        g_start = start - (resampled_idx - window)
        g_end = g_start + (end - start)
        heatmaps[i, start:end] = torch.from_numpy(gaussian[g_start:g_end]).float()

    return heatmaps
