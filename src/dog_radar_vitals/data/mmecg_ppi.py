"""radarODE(Algorithm 1)に基づく、レーダのみからのPPI(peak-to-peak interval、拍間隔)推定。

ECGを一切参照せず、50chのSSTエネルギープロット（周波数軸で積分した1次元信号）だけから
各チャネル独立にピーク検出し、候補PPI値をKDEで多数決することで頑健にPPIを推定する
（1チャネルだけでは低SNRでピーク検出に失敗しうるが、50チャネルの大多数は正しく検出できる、
という前提に基づく）。推定したPPIは、`mmecg_singlecycle_dataset.py`で1心拍単位に
レーダ信号・ECGを切り出す境界の決定に使う（学習時・推論時とも同じ手順=レーダのみに基づく
ため、ECGの事前知識によるリークは無い）。

論文Algorithm 1の完全な再現ではなく、多チャネル多数決というアイデアの核心を保った簡略実装
（NeuroKit2のbiopeaksではなくscipy.find_peaksを使う、KDEはセグメント単位ではなく
スライディングウィンドウ単位、など）。詳細な相違点は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照。

## 2026-07-29追記: `sliding_ppi_estimates`+`local_ppi_sec`の開ループ的ウォークが持つ位相ドリフト

`estimate_cycle_bounds`(`mmecg_singlecycle_dataset.py`)は、当初t=0から「局所PPI推定値の
分だけ前進する」ことを繰り返して拍境界を決めていた。これは各ステップのPPI推定誤差が
補正されずに蓄積する開ループ制御に相当し、実際に検証したところ(trial 1)、真のR波位置との
乖離が最大±0.46秒（ほぼ半心拍分）に達する区間があった。加えて検出拍数も231(推定) vs
201(真のR波数、参考としてのみ算出)と15%過剰検出していた。

これを解消するため、`detect_consensus_beat_times`を追加した。局所PPI"値"を推定して歩くの
ではなく、50チャネル全体で検出された候補ピーク"時刻"そのものをプールし、ガウシアンカーネルで
時間軸上に積み上げた「スパイク密度関数」の極大点を拍タイミングとして直接採用する
（神経科学のスパイク列解析で使われるKernel Density Estimation of point processと同じ発想）。
これにより、個々の拍の境界が実際に検出されたピーク位置に直接アンカーされ、開ループ的な
誤差蓄積が起きない。`mmecg_singlecycle_dataset.py`はこの関数の出力(拍タイミング)の中点を
拍境界として使う。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

MIN_PPI_SEC = 0.3  # 生理的に妥当な最小拍間隔(=最大200bpm相当、rpeaks.pyのMIN_RR_SECと同じ)
MAX_PPI_SEC = 2.0  # 最小30bpm相当
BEAT_KERNEL_SEC = 0.15  # 拍タイミングのコンセンサス検出に使うガウシアンカーネルの標準偏差
MIN_VOTES_FRAC = 0.06  # 50chのうち何割が同じ時刻付近でピークを検出すれば「拍」と認めるか
# 実データ(trial 1/2/5/9/10)でグリッドサーチし、検出拍数が真のR波数に最も近い値を選んだ
# (`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「タスク設計・前処理・
# 学習設計の改善」節参照)。


def energy_plot(sst_magnitude: np.ndarray) -> np.ndarray:
    """SSTスペクトログラム(F, T)を周波数軸で積分し、1次元のエネルギープロット(T,)を返す。"""
    return sst_magnitude.sum(axis=0)


def _channel_candidate_ppis(energy: np.ndarray, fs: float) -> np.ndarray:
    """1チャネルのエネルギープロットからピーク検出し、隣接ピーク間隔(秒)の候補を返す。"""
    z = (energy - energy.mean()) / (energy.std() + 1e-8)
    peaks, _ = find_peaks(z, height=0.5, distance=max(1, int(MIN_PPI_SEC * fs)))
    if len(peaks) < 2:
        return np.array([])
    ppi = np.diff(peaks) / fs
    return ppi[(ppi >= MIN_PPI_SEC) & (ppi <= MAX_PPI_SEC)]


def estimate_ppi_sec(sst_50ch: np.ndarray, fs: float) -> float:
    """50ch分のSST(50, F, T)から、多数決でこの区間の代表PPI(秒)を1つ推定する。

    全チャネルの候補PPIをプールし、KDEで最頻値を採用する（1〜数チャネルの誤検出に
    引っ張られないようにする）。候補が少なすぎる場合は生理的に妥当な既定値(1.0秒=60bpm)を返す。
    """
    all_candidates = []
    for ch in range(sst_50ch.shape[0]):
        energy = energy_plot(sst_50ch[ch])
        all_candidates.append(_channel_candidate_ppis(energy, fs))
    candidates = np.concatenate(all_candidates) if all_candidates else np.array([])

    if len(candidates) < 5:
        return 1.0

    kde = gaussian_kde(candidates, bw_method="silverman")
    grid = np.linspace(MIN_PPI_SEC, MAX_PPI_SEC, 200)
    density = kde(grid)
    return float(grid[np.argmax(density)])


def sliding_ppi_estimates(sst_50ch: np.ndarray, fs: float, segment_sec: float = 10.0, step_sec: float = 5.0) -> list[tuple[float, float]]:
    """SST全体をスライディングウィンドウでPPI推定し、(区間中心時刻[秒], PPI[秒])のリストを返す。

    `mmecg_singlecycle_dataset.py`は、この局所PPI推定値を時刻に応じて補間しながら、
    連続する1心拍分の境界を順に切り出していく。
    """
    total_t = sst_50ch.shape[-1]
    seg_len = int(segment_sec * fs)
    step_len = int(step_sec * fs)

    estimates = []
    for start in range(0, max(total_t - seg_len, 1), step_len):
        end = min(start + seg_len, total_t)
        ppi = estimate_ppi_sec(sst_50ch[:, :, start:end], fs)
        center_sec = (start + end) / 2 / fs
        estimates.append((center_sec, ppi))

    if not estimates:
        estimates = [(total_t / 2 / fs, estimate_ppi_sec(sst_50ch, fs))]
    return estimates


def local_ppi_sec(estimates: list[tuple[float, float]], t_sec: float) -> float:
    """`sliding_ppi_estimates`の結果を時刻`t_sec`に応じて線形補間する。"""
    centers = np.array([c for c, _ in estimates])
    ppis = np.array([p for _, p in estimates])
    if len(centers) == 1:
        return float(ppis[0])
    return float(np.interp(t_sec, centers, ppis))


def _all_channel_peak_times_sec(sst_50ch: np.ndarray, fs: float) -> np.ndarray:
    """全チャネルの候補ピーク時刻(秒)をプールして返す(チャネル間の対応関係は捨てる)。"""
    peak_times = []
    for ch in range(sst_50ch.shape[0]):
        energy = energy_plot(sst_50ch[ch])
        z = (energy - energy.mean()) / (energy.std() + 1e-8)
        peaks, _ = find_peaks(z, height=0.5, distance=max(1, int(MIN_PPI_SEC * fs)))
        peak_times.append(peaks / fs)
    return np.concatenate(peak_times) if peak_times else np.array([])


def detect_consensus_beat_times(sst_50ch: np.ndarray, fs: float) -> np.ndarray:
    """50chの候補ピーク時刻の多数決から、拍タイミング(秒、昇順)を直接検出する。

    `sliding_ppi_estimates`+`local_ppi_sec`による開ループウォーク(誤差が蓄積し実際の
    R波位置から系統的にズレる、モジュールdocstring参照)とは異なり、実際に検出された
    候補ピーク時刻をガウシアンカーネルで積み上げたスパイク密度関数の極大点を拍タイミング
    として採用するため、各拍が実際のピーク位置に直接アンカーされる。
    """
    n_channels = sst_50ch.shape[0]
    total_t = sst_50ch.shape[-1]
    duration_sec = total_t / fs
    peak_times = _all_channel_peak_times_sec(sst_50ch, fs)
    if len(peak_times) < 5:
        return np.array([])

    grid_len = max(int(round(duration_sec * fs)), 1)
    hist = np.zeros(grid_len, dtype=np.float64)
    idx = np.clip((peak_times * fs).astype(int), 0, grid_len - 1)
    np.add.at(hist, idx, 1.0)

    sigma_samples = BEAT_KERNEL_SEC * fs
    density = gaussian_filter1d(hist, sigma=sigma_samples)

    kernel_peak = 1.0 / (sigma_samples * np.sqrt(2 * np.pi))
    min_height = MIN_VOTES_FRAC * n_channels * kernel_peak
    peak_idx, _ = find_peaks(density, height=min_height, distance=max(1, int(MIN_PPI_SEC * fs)))
    return peak_idx / fs
