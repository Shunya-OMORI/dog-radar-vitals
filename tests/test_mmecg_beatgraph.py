from pathlib import Path

import numpy as np
import pytest
import torch

from dog_radar_vitals.data.mmecg import MMECGRecording
from dog_radar_vitals.data.mmecg_beatgraph_dataset import iter_beat_graphs
from dog_radar_vitals.models.deep.beatgraph_discriminator import BeatGraphDiscriminator
from dog_radar_vitals.models.deep.beatgraph_gnn import BeatGraphGNN

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
HAS_DATA = (RAW_ROOT / "mmecg").exists()


def _synthetic_ecg_recording(n_beats: int = 20, fs: int = 200) -> MMECGRecording:
    """P-QRS-T形状に似た周期波形を合成し、neurokit2のdelineationが通る程度のECGを作る。"""
    rr_samples = int(0.8 * fs)
    n = n_beats * rr_samples + rr_samples
    t = np.arange(n) / fs
    ecg = np.zeros(n)
    for beat in range(n_beats + 1):
        center = beat * rr_samples
        for offset, amp, width in [(-40, 0.15, 8), (-15, -0.1, 3), (0, 1.0, 4), (15, -0.25, 3), (60, 0.3, 15)]:
            idx = center + offset
            if 0 <= idx < n:
                span = np.arange(max(0, idx - 3 * width), min(n, idx + 3 * width))
                ecg[span] += amp * np.exp(-0.5 * ((span - idx) / width) ** 2)
    rng = np.random.default_rng(0)
    rcg = rng.normal(size=(n, 50)).astype(np.float32)
    posxyz = rng.normal(size=(50, 3)).astype(np.float32)
    return MMECGRecording(trial_id=0, subject_id=0, fs=fs, rcg=rcg, ecg=ecg.astype(np.float32), posxyz=posxyz, physistatus="NB", age=30, gender="boy")


def test_iter_beat_graphs_on_synthetic_ecg():
    rec = _synthetic_ecg_recording()
    samples = iter_beat_graphs(rec, segment_sec=0.4)
    assert len(samples) > 0
    for rcg_seg, graph in samples:
        assert rcg_seg.shape == (80, 50)
        assert graph.shape == (5, 2)
        assert not np.isnan(graph).any()


def test_beatgraph_gnn_forward_shape():
    model = BeatGraphGNN(in_channels=50, embed_dim=16, gat_hidden=16, n_gat_layers=2)
    x = torch.randn(4, 80, 50)
    y = model(x)
    assert y.shape == (4, 5, 2)


def test_beatgraph_discriminator_forward_shape():
    disc = BeatGraphDiscriminator()
    graph = torch.randn(4, 5, 2)
    logit = disc(graph)
    assert logit.shape == (4,)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw/mmecg が未配置")
def test_mmecg_beatgraph_dataset_on_real_trial():
    from dog_radar_vitals.data.mmecg_beatgraph_dataset import MMECGBeatGraphDataset

    ds = MMECGBeatGraphDataset(RAW_ROOT, [1], segment_sec=0.8)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (160, 50)
    assert y.shape == (5, 2)
