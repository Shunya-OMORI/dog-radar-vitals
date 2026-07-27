from pathlib import Path

import numpy as np
import pytest
import torch

from dog_radar_vitals.data.mmecg import MMECGRecording, list_trials, load_trial, trials_by_subject
from dog_radar_vitals.data.mmecg_dataset import MMECGWindowDataset
from dog_radar_vitals.data.mmecg_rpeak_dataset import MMECGRPeakWindowDataset
from dog_radar_vitals.data.mmecg_rpeak_windowing import iter_peak_windows
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset
from dog_radar_vitals.data.mmecg_windowing import iter_windows

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
HAS_DATA = (RAW_ROOT / "mmecg").exists()


def _fake_recording(n: int = 1000, fs: int = 200, n_channels: int = 50) -> MMECGRecording:
    rng = np.random.default_rng(0)
    rcg = rng.normal(size=(n, n_channels)).astype(np.float32)
    t = np.arange(n) / fs
    ecg = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)
    posxyz = rng.normal(size=(n_channels, 3)).astype(np.float32)
    return MMECGRecording(trial_id=0, subject_id=0, fs=fs, rcg=rcg, ecg=ecg, posxyz=posxyz, physistatus="NB", age=30, gender="boy")


def test_iter_windows_shapes_and_skips_nan():
    rec = _fake_recording()
    rec.ecg[500:505] = np.nan
    windows = list(iter_windows(rec, window_sec=1, stride_sec=1))
    assert len(windows) > 0
    for rcg_win, ecg_win in windows:
        assert rcg_win.shape == (200, 50)
        assert ecg_win.shape == (200,)
        assert not np.isnan(ecg_win).any()


def test_iter_windows_complex_input_produces_complex64():
    rec = _fake_recording()
    rcg_win, _ = next(iter_windows(rec, window_sec=1, stride_sec=1, complex_input=True))
    assert rcg_win.dtype == np.complex64
    assert rcg_win.shape == (200, 50)


def test_iter_peak_windows_shapes():
    rec = _fake_recording(n=2000)
    windows = list(iter_peak_windows(rec, window_sec=1, stride_sec=1))
    assert len(windows) > 0
    for rcg_win, heatmap_win in windows:
        assert rcg_win.shape == (200, 50)
        assert heatmap_win.shape == (200,)
        assert heatmap_win.min() >= 0.0 and heatmap_win.max() <= 1.0


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_list_trials_finds_ninety_one():
    trials = list_trials(RAW_ROOT)
    assert len(trials) == 91


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_load_trial_shapes():
    rec = load_trial(RAW_ROOT, 1)
    assert rec.fs == 200
    assert rec.rcg.shape == (35505, 50)
    assert rec.ecg.shape == (35505,)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_trials_by_subject_has_eleven_subjects():
    mapping = trials_by_subject(RAW_ROOT)
    assert len(mapping) == 11
    assert sum(len(v) for v in mapping.values()) == 91


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_mmecg_window_dataset_shapes():
    ds = MMECGWindowDataset(RAW_ROOT, [1], window_sec=4, stride_sec=2)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (800, 50)
    assert y.shape == (800,)
    assert x.dtype == torch.float32


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_mmecg_rpeak_dataset_shapes():
    ds = MMECGRPeakWindowDataset(RAW_ROOT, [1], window_sec=4, stride_sec=2)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (800, 50)
    assert y.shape == (800,)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_mmecg_spatial_window_dataset_shapes():
    ds = MMECGSpatialWindowDataset(RAW_ROOT, [1], window_sec=4, stride_sec=2)
    assert len(ds) > 0
    rcg, posxyz, ecg = ds[0]
    assert rcg.shape == (800, 50)
    assert posxyz.shape == (50, 3)
    assert ecg.shape == (800,)
