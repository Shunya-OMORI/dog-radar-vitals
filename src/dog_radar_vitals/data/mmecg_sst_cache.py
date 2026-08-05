"""RCG(50ch)全チャネルのSST変換を事前計算し、`data/processed/mmecg_sst*/`にキャッシュする。

SST計算は1ch・1トライアル(3分,35505サンプル)あたり約0.7秒かかり、50ch×91トライアルを
毎エポック再計算すると現実的な時間で終わらない。学習・評価のたびに再計算しないよう、
トライアル単位で計算結果をnpzキャッシュする（`README.md`の設計方針「データローディング・
前処理は`data/processed/`にキャッシュ」に対応する、本リポジトリで`data/processed/`を
初めて実利用する箇所）。

周波数・時間の両軸をダウンサンプルして保存サイズを抑える:
- 周波数: [1, 25]Hz帯域のSST出力(約150bin)を、`N_FREQ_BINS`個へ等間隔に間引く。
- 時間: 2サンプルに1つを残す(200Hz→100Hz相当。SSTは[1,25]Hzのみを保持するので
  Nyquist(50Hz)に対して十分余裕がある)。

## バンドパス+正規化の見直し(2026-07-29)

従来は生の広帯域RCGに対し直接z-score正規化してからSST変換していた。実測したところ、
この正規化の分散計算が体動由来と推測される25Hz超の高周波ノイズに支配されており
(trial 1 ch0で、10秒窓ごとのstdが生広帯域では最大9.3倍ばらつくのに対し、[1,25]Hz
バンドパス後は3.4倍まで縮小、詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
参照)、正規化前に`bandpass.py`で心拍帯[1,25]Hzのバンドパスフィルタをかけるよう変更した。
既存のキャッシュ(`mmecg_sst/`、フィルタなし)とは前処理が異なるため、混同を避けるため
別ディレクトリ(`mmecg_sst_bp/`)にキャッシュする。`apply_bandpass=False`で従来の
フィルタなし版(`mmecg_sst/`)も引き続き使える。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from dog_radar_vitals.data.bandpass import bandpass_filter
from dog_radar_vitals.data.mmecg import MMECGRecording, load_trial
from dog_radar_vitals.data.sst import sst_magnitude

N_FREQ_BINS = 64
TIME_DECIMATION = 2
CACHE_DIRNAME = "mmecg_sst"
CACHE_DIRNAME_BANDPASS = "mmecg_sst_bp"


def _cache_path(raw_root: Path, trial_id: int, apply_bandpass: bool) -> Path:
    dirname = CACHE_DIRNAME_BANDPASS if apply_bandpass else CACHE_DIRNAME
    return Path(raw_root).parent / "processed" / dirname / f"{trial_id}.npz"


def compute_sst_all_channels(rec: MMECGRecording, apply_bandpass: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """RCG(T, 50)の全チャネルをSST変換する。-> (sst(50, N_FREQ_BINS, T//2), freqs(N_FREQ_BINS,))。

    `apply_bandpass=True`の場合、z-score正規化の前に[1,25]Hzバンドパスフィルタをかけ、
    高周波ノイズに支配されない正規化統計量を使う(モジュールdocstring参照)。
    """
    n_channels = rec.rcg.shape[1]
    channel_ssts = []
    freqs_ref = None
    for ch in range(n_channels):
        x = rec.rcg[:, ch].astype(np.float64)
        x = np.nan_to_num(x, nan=np.nanmean(x))
        if apply_bandpass:
            x = bandpass_filter(x, rec.fs)
        x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)
        mag, freqs = sst_magnitude(x, rec.fs)

        # 等間隔に間引いてN_FREQ_BINS本に揃える(トライアル間で周波数軸の形状を一致させる)。
        idx = np.linspace(0, len(freqs) - 1, N_FREQ_BINS).round().astype(int)
        mag = mag[idx][:, ::TIME_DECIMATION]
        if freqs_ref is None:
            freqs_ref = freqs[idx]
        channel_ssts.append(mag)

    return np.stack(channel_ssts, axis=0), freqs_ref


def get_or_compute_sst(raw_root: Path, trial_id: int, apply_bandpass: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """トライアルのSST(50ch)をキャッシュから読むか、無ければ計算してキャッシュする。"""
    path = _cache_path(raw_root, trial_id, apply_bandpass)
    if path.exists():
        data = np.load(path)
        return data["sst"], data["freqs"]

    rec = load_trial(raw_root, trial_id)
    sst, freqs = compute_sst_all_channels(rec, apply_bandpass=apply_bandpass)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, sst=sst.astype(np.float32), freqs=freqs.astype(np.float32))
    return sst, freqs
