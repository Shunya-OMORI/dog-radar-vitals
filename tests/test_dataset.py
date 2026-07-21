from pathlib import Path

import numpy as np
import pytest

from dog_radar_vitals.data.dataset import WindowedVitalsDataset
from dog_radar_vitals.data.features import FEATURE_NAMES, build_feature_table, extract_window_features
from dog_radar_vitals.data.scenario1 import list_dogs, load_recording
from dog_radar_vitals.data.windowing import zscore

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
HAS_DATA = (RAW_ROOT / "Scenario1_FaintedDogs").exists()


def test_zscore_normalizes_per_column():
    x = np.array([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0]], dtype=np.float32)
    z = zscore(x)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z[:, 1], 0.0, atol=1e-3)  # 分散ゼロの列でもゼロ割しない


def test_extract_window_features_shape():
    window = np.random.randn(500, 467).astype(np.float32)
    features = extract_window_features(window)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


@pytest.mark.skipif(not HAS_DATA, reason="data/raw にデータセットが未配置")
def test_list_dogs_finds_ten_recordings():
    dogs = list_dogs(RAW_ROOT)
    assert len(dogs) == 10
    assert dogs[0] == "No1"


@pytest.mark.skipif(not HAS_DATA, reason="data/raw にデータセットが未配置")
def test_load_recording_shapes():
    rec = load_recording(RAW_ROOT, "No1")
    assert rec.radar.shape == (9000, 467)
    assert rec.hr.shape == (180,)
    assert rec.br.shape == (180,)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw にデータセットが未配置")
def test_windowed_dataset_shapes_and_alignment():
    ds = WindowedVitalsDataset(RAW_ROOT, ["No1"], task="hr", window_sec=10, stride_sec=1)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (500, 467)  # 10秒 * 50Hz
    assert y.ndim == 0


@pytest.mark.skipif(not HAS_DATA, reason="data/raw にデータセットが未配置")
def test_build_feature_table_matches_dataset_length():
    ds = WindowedVitalsDataset(RAW_ROOT, ["No1"], task="hr", window_sec=10, stride_sec=1)
    X, y = build_feature_table(RAW_ROOT, ["No1"], "hr", window_sec=10, stride_sec=1)
    assert X.shape == (len(ds), len(FEATURE_NAMES))
    assert y.shape == (len(ds),)
