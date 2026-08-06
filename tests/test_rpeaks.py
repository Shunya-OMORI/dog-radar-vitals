import numpy as np
import pytest

from dog_radar_vitals.data.rpeaks import (
    build_peak_heatmap,
    detect_r_peaks,
    extract_peaks_adaptive_searchback,
    extract_peaks_from_heatmap,
    extract_peaks_paired_dedup,
    extract_peaks_rhythmic,
    match_peaks,
    matched_rr_interval_mae_ms,
    rr_intervals_ms,
)
from dog_radar_vitals.models.deep.rpeak_cnn1d import RPeakCNN1D
import torch


def _synthetic_ecg(fs: int = 2000, duration_sec: float = 10.0, hr_bpm: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    n = int(fs * duration_sec)
    rr_sec = 60.0 / hr_bpm
    peak_times = np.arange(rr_sec, duration_sec, rr_sec)
    peak_indices = (peak_times * fs).astype(int)

    ecg = np.random.randn(n).astype(np.float32) * 0.05
    for idx in peak_indices:
        width = 20
        t = np.arange(-width, width)
        spike = 5.0 * np.exp(-0.5 * (t / 5.0) ** 2)
        start, end = max(0, idx - width), min(n, idx + width)
        ecg[start:end] += spike[: end - start]
    return ecg, peak_indices


def test_detect_r_peaks_recovers_synthetic_beats():
    ecg, true_peaks = _synthetic_ecg()
    detected = detect_r_peaks(ecg, fs=2000)
    assert len(detected) == len(true_peaks)
    assert np.abs(detected - true_peaks).max() < 5  # サンプル単位でほぼ一致


def test_rr_intervals_ms_matches_known_hr():
    ecg, true_peaks = _synthetic_ecg(hr_bpm=60.0)
    detected = detect_r_peaks(ecg, fs=2000)
    rr = rr_intervals_ms(detected, fs=2000)
    assert np.allclose(rr, 1000.0, atol=5)  # 60bpm -> RR=1000ms


def test_build_and_extract_heatmap_roundtrip():
    # 隣接ピーク間隔は最小RR間隔(300ms=600サンプル@2000Hz)より十分離す
    peak_indices = np.array([100, 900, 1700])
    heatmap = build_peak_heatmap(peak_indices, length=2000, fs=2000, sigma_ms=10)
    assert heatmap.max() <= 1.0 and heatmap.min() >= 0.0

    extracted = extract_peaks_from_heatmap(heatmap, fs=2000, height=0.3)
    assert len(extracted) == len(peak_indices)
    assert np.abs(np.sort(extracted) - peak_indices).max() < 3


def test_match_peaks_perfect_detection():
    true_peaks = np.array([1000, 3000, 5000, 7000])
    pred_peaks = true_peaks + 5  # 5サンプル(2.5ms)のずれ、許容誤差内
    result = match_peaks(true_peaks, pred_peaks, fs=2000, tolerance_ms=50)
    assert result["n_matched"] == 4
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    assert np.allclose(result["timing_errors_ms"], 2.5)


def test_match_peaks_with_missed_and_extra_detections():
    true_peaks = np.array([1000, 3000, 5000, 7000])
    pred_peaks = np.array([1000, 5000, 7000, 9000])  # 3000を見逃し、9000を過検出
    result = match_peaks(true_peaks, pred_peaks, fs=2000, tolerance_ms=50)
    assert result["n_matched"] == 3
    assert result["precision"] == 3 / 4
    assert result["recall"] == 3 / 4


def test_matched_rr_interval_mae_ms():
    true_peaks = np.array([0, 2000, 4000, 6000])  # RR=1000ms一定(fs=2000)
    pred_peaks = np.array([0, 2100, 3900, 6000])  # RRが1050,900,1050msにずれる
    result = match_peaks(true_peaks, pred_peaks, fs=2000, tolerance_ms=100)
    mae = matched_rr_interval_mae_ms(result, fs=2000)
    assert mae is not None
    # true_rr=[1000,1000,1000]ms, pred_rr=[1050,900,1050]ms -> |diff|=[50,100,50] -> mean=200/3
    assert mae == pytest.approx(200 / 3, abs=1e-6)


def test_extract_peaks_paired_dedup_merges_close_rt_like_pair():
    # R(idx=1000)の370ms後(fs=2000なので740サンプル後)にT波相当の小さい山がある
    # ケースを合成する。pair_merge_sec=0.45秒(900サンプル)より近いので後者は除外される。
    fs = 2000
    length = 4000
    heatmap = np.zeros(length, dtype=np.float32)
    r_idx, t_idx = 1000, 1740
    width = 40
    t_axis = np.arange(-width, width)
    heatmap[r_idx - width : r_idx + width] += np.exp(-0.5 * (t_axis / 10.0) ** 2)
    heatmap[t_idx - width : t_idx + width] += 0.5 * np.exp(-0.5 * (t_axis / 10.0) ** 2)

    old = extract_peaks_from_heatmap(heatmap, fs=fs, height=0.10)
    assert len(old) == 2  # 旧方式(不応期0.3秒)ではR/Tとも検出されてしまう

    new = extract_peaks_paired_dedup(heatmap, fs=fs, height=0.10, pair_merge_sec=0.45)
    assert len(new) == 1
    assert abs(int(new[0]) - r_idx) < 5


def test_extract_peaks_paired_dedup_keeps_genuinely_spaced_beats():
    # 900ms間隔(pair_merge_secの0.45秒より十分長い)の通常のRR間隔は統合されない
    fs = 2000
    length = 4000
    heatmap = np.zeros(length, dtype=np.float32)
    peak_indices = [500, 2300]  # 900ms間隔
    width = 40
    t_axis = np.arange(-width, width)
    for idx in peak_indices:
        heatmap[idx - width : idx + width] += np.exp(-0.5 * (t_axis / 10.0) ** 2)

    new = extract_peaks_paired_dedup(heatmap, fs=fs, height=0.10, pair_merge_sec=0.45)
    assert len(new) == 2


def _bump(heatmap: np.ndarray, idx: int, amplitude: float, width: int = 40, sigma: float = 10.0) -> None:
    t_axis = np.arange(-width, width)
    lo, hi = max(0, idx - width), min(len(heatmap), idx + width)
    heatmap[lo:hi] += amplitude * np.exp(-0.5 * (t_axis[: hi - lo] / sigma) ** 2)


def test_extract_peaks_rhythmic_rejects_irregular_noise_spikes():
    # 一定間隔(600サンプル=300ms@2000Hz)の規則的な系列に、間隔が不規則な雑音ピークを混ぜる。
    fs = 2000
    length = 6000
    heatmap = np.zeros(length, dtype=np.float32)
    regular_idx = [500, 1100, 1700, 2300, 2900, 3500, 4100, 4700, 5300]
    for idx in regular_idx:
        _bump(heatmap, idx, amplitude=1.0)
    noise_idx = [850, 2050, 3950]  # 規則的な系列とは無関係な位置
    for idx in noise_idx:
        _bump(heatmap, idx, amplitude=0.6)

    extracted = extract_peaks_rhythmic(heatmap, fs=fs, candidate_height=0.05, min_rr_sec=0.2, max_rr_sec=0.5)
    assert len(extracted) == len(regular_idx)
    assert np.abs(np.sort(extracted) - np.array(regular_idx)).max() < 5


def test_extract_peaks_rhythmic_prefers_higher_amplitude_regular_chain_over_rt_like_pair():
    # 各周期でR相当(高振幅)とT相当(低振幅・R波の370ms後)の2つの規則的な系列が並走する場合、
    # 高振幅側(R)の系列が選ばれることを確認する。
    fs = 2000
    length = 6000
    heatmap = np.zeros(length, dtype=np.float32)
    r_idx = [500, 1500, 2500, 3500, 4500]
    t_idx = [i + 740 for i in r_idx]  # 370ms後
    for idx in r_idx:
        _bump(heatmap, idx, amplitude=1.0)
    for idx in t_idx:
        _bump(heatmap, idx, amplitude=0.5)

    extracted = extract_peaks_rhythmic(heatmap, fs=fs, candidate_height=0.05, min_rr_sec=0.2, max_rr_sec=1.5)
    assert len(extracted) == len(r_idx)
    assert np.abs(np.sort(extracted) - np.array(r_idx)).max() < 5


def test_extract_peaks_adaptive_searchback_recovers_regular_beats_with_noise():
    fs = 2000
    length = 6000
    heatmap = np.zeros(length, dtype=np.float32)
    regular_idx = [500, 1100, 1700, 2300, 2900, 3500, 4100, 4700, 5300]
    for idx in regular_idx:
        _bump(heatmap, idx, amplitude=1.0)
    noise_idx = [850, 2050, 3950]
    for idx in noise_idx:
        _bump(heatmap, idx, amplitude=0.2)  # 十分弱いノイズ = ノイズしきい値を超えない想定

    extracted = extract_peaks_adaptive_searchback(heatmap, fs=fs, candidate_height=0.05, init_rr_sec=0.3)
    assert len(extracted) == len(regular_idx)
    assert np.abs(np.sort(extracted) - np.array(regular_idx)).max() < 5


def test_extract_peaks_adaptive_searchback_recovers_missed_beat_via_searchback():
    # 1周期分(idx=1700相当)だけ振幅が弱く、通常しきい値では見逃されるが、
    # searchbackの緩和しきい値では拾えるケース。
    fs = 2000
    length = 6000
    heatmap = np.zeros(length, dtype=np.float32)
    strong_idx = [500, 1100, 2300, 2900, 3500]
    weak_idx = 1700  # 見逃されそうな弱いR波
    for idx in strong_idx:
        _bump(heatmap, idx, amplitude=1.0)
    _bump(heatmap, weak_idx, amplitude=0.35)

    extracted = extract_peaks_adaptive_searchback(heatmap, fs=fs, candidate_height=0.05, init_rr_sec=0.3)
    assert len(extracted) == len(strong_idx) + 1
    assert np.any(np.abs(np.sort(extracted) - weak_idx) < 5)


def test_rpeak_cnn1d_forward_shape_and_range():
    model = RPeakCNN1D(channels=16, n_layers=3, kernel_size=5)
    x = torch.randn(2, 400, 2)
    y = model(x)
    assert y.shape == (2, 400)
    assert (y >= 0).all() and (y <= 1).all()
