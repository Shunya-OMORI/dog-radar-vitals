import numpy as np

from dog_radar_vitals.data.bandpass import bandpass_filter, ema_clutter_removal


def test_bandpass_filter_removes_dc_offset():
    fs = 200
    t = np.arange(fs * 4) / fs
    x = 5.0 + np.sin(2 * np.pi * 3.0 * t)  # DCオフセット5 + 3Hz成分(通過帯域内)
    y = bandpass_filter(x, fs)
    assert abs(y.mean()) < 0.5  # DC成分は除去される


def test_ema_clutter_removal_removes_slow_trend():
    x = np.linspace(0, 10, 2000) + np.random.randn(2000) * 0.01
    y = ema_clutter_removal(x, alpha=0.01)
    # 緩やかなトレンドは基線として追従・除去され、先頭と末尾の平均がゼロに近づく
    assert abs(y[-200:].mean()) < abs(x[-200:].mean() - x[:200].mean()) * 0.5


def test_ema_clutter_removal_preserves_fast_transient():
    # 緩やかなトレンド + 短い急峻なパルス(QRS相当)。パルスは基線に追従されず残るはず。
    n = 1000
    x = np.linspace(0, 2, n)
    pulse_idx = 500
    x[pulse_idx - 5:pulse_idx + 5] += 3.0
    y = ema_clutter_removal(x, alpha=0.01)
    assert y[pulse_idx] > 1.0  # パルスの高さはほぼ保たれる
