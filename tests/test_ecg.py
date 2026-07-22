from pathlib import Path

import numpy as np
import pytest
import torch

from dog_radar_vitals.data.ecg_dataset import ECGWindowDataset
from dog_radar_vitals.data.ecg_windowing import iter_windows, zscore_with_nan_gap
from dog_radar_vitals.data.schellenberger import HumanRecording, list_subjects, load_recording
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
HAS_DATA = (RAW_ROOT / "schellenberger_human").exists()


def test_zscore_with_nan_gap_does_not_poison_whole_array():
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0], dtype=np.float32)
    z = zscore_with_nan_gap(x)
    assert np.isnan(z[3])
    assert not np.isnan(z[[0, 1, 2, 4, 5]]).any()


def test_ecg_cnn1d_forward_shape():
    model = build_ecg_model("ecg_cnn1d", channels=16, n_layers=3, kernel_size=5)
    x = torch.randn(2, 400, 2)
    y = model(x)
    assert y.shape == (2, 400)


def test_iter_windows_skips_windows_overlapping_nan():
    fs = 100
    n = 1000
    ecg = np.zeros(n, dtype=np.float32)
    ecg[500:505] = np.nan
    rec = HumanRecording(subject_id="X", scenario="test", fs=fs, radar_i=np.zeros(n, dtype=np.float32), radar_q=np.zeros(n, dtype=np.float32), ecg=ecg)
    windows = list(iter_windows(rec, window_sec=1, stride_sec=1))
    for _, ecg_win in windows:
        assert not np.isnan(ecg_win).any()


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/schellenberger_human が未配置")
def test_list_subjects_finds_ten():
    subjects = list_subjects(RAW_ROOT)
    assert len(subjects) == 10
    assert subjects[0] == "GDN0001"


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/schellenberger_human が未配置")
def test_load_recording_radar_ecg_same_length_and_fs():
    rec = load_recording(RAW_ROOT, "GDN0001", "Resting")
    assert rec.fs == 2000
    assert len(rec.radar_i) == len(rec.radar_q) == len(rec.ecg)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/schellenberger_human が未配置")
def test_ecg_window_dataset_shapes():
    ds = ECGWindowDataset(RAW_ROOT, ["GDN0001"], "Resting", window_sec=4, stride_sec=2)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (8000, 2)
    assert y.shape == (8000,)
