import numpy as np
import pytest

from dog_radar_vitals.data.rpeaks import (
    build_peak_heatmap,
    detect_r_peaks,
    extract_peaks_from_heatmap,
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


def test_rpeak_cnn1d_forward_shape_and_range():
    model = RPeakCNN1D(channels=16, n_layers=3, kernel_size=5)
    x = torch.randn(2, 400, 2)
    y = model(x)
    assert y.shape == (2, 400)
    assert (y >= 0).all() and (y <= 1).all()
