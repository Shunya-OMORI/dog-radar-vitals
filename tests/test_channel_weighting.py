import numpy as np

from dog_radar_vitals.data.channel_weighting import (
    apply_channel_weights,
    cardiac_band_power_ratio,
    compute_channel_weights,
)


def test_cardiac_band_power_ratio_higher_for_heartbeat_like_signal():
    fs = 200
    t = np.arange(fs * 8) / fs
    heartbeat_like = np.sin(2 * np.pi * 1.2 * t)  # 心拍帯(72bpm相当)の純音
    noise_like = np.random.randn(len(t)) * 0.1 + np.sin(2 * np.pi * 40 * t)  # 高周波ノイズ主体

    r_heart = cardiac_band_power_ratio(heartbeat_like, fs)
    r_noise = cardiac_band_power_ratio(noise_like, fs)
    assert r_heart > r_noise


def test_compute_channel_weights_ranks_heartbeat_channel_highest():
    fs = 200
    n = fs * 8
    t = np.arange(n) / fs
    rcg = np.zeros((n, 3))
    rcg[:, 0] = np.sin(2 * np.pi * 1.2 * t)  # 心拍帯
    rcg[:, 1] = np.random.randn(n) * 0.1 + np.sin(2 * np.pi * 40 * t)  # 高周波ノイズ
    rcg[:, 2] = np.random.randn(n) * 0.01  # ほぼ無信号

    weights = compute_channel_weights(rcg, fs)
    assert weights[0] == 1.0  # 最大は正規化で1.0
    assert weights[0] > weights[1]
    assert weights[0] > weights[2]


def test_apply_channel_weights_scales_correctly():
    rcg = np.ones((10, 3))
    weights = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    out = apply_channel_weights(rcg, weights)
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 1], 0.5)
    assert np.allclose(out[:, 2], 0.0)
