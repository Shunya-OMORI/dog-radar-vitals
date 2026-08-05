"""radarODEのSCEG(Single-Cycle ECG Generator)学習用: 1心拍単位のSSTセグメント→ECG波形Dataset。

`mmecg_ppi.py`で推定したPPI(レーダのみに基づく、ECGを参照しない)で、SST全体を連続する
1心拍分の区間へ順に区切っていく。各区間について:
- 入力: その区間のSST(50ch)を、時間軸方向に固定長`T_FIXED_SST`へ線形補間でリサイズしたもの。
- 正解: 対応するECG区間(200Hzの生値)を、固定長`T_FIXED_ECG`(原著と同じ200)へ
  `scipy.signal.resample`でリサンプルしたもの。

## 遅延ロード方式(2026-07-28、メモリ制約下でのT_FIXED_SST復元)

`__init__`ではトライアルごとの拍境界(`(trial_id, sst_start, sst_end, ecg_start,
ecg_end)`)という軽量なタプルのみを保持し、実際のSSTセグメント切り出し・リサイズは
`__getitem__`で遅延評価する。トライアル単位のSST全体は`_TrialSSTCache`という固定サイズ
LRUキャッシュで保持し、`TrialInterleavedSampler`で局所性を保ったシャッフルを行う
(詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「radarODEの実装・
再現結果」節)。

## 拍境界推定の見直し(2026-07-29、タスク設計の改善)

`mmecg_ppi.detect_consensus_beat_times`(50chの候補ピーク時刻をガウシアンカーネルで
積み上げたスパイク密度関数の極大点を拍タイミングとして直接採用)を使い、検出された拍タイミング
の中点を境界とする。旧PPIウォーク方式(開ループで誤差蓄積、最大±0.46秒の位相ドリフト実測)
より頑健。候補ピークが少なすぎる場合のみ`_compute_bounds_legacy`にフォールバックする。

## 4秒文脈窓は悪化したため既定OFF(2026-07-29)

radarODE論文本文(p.6)の"the actual SST segment is centered at the current cardiac cycle
and expands to 4 seconds"という記述に基づき`use_context_window=True`を試したが、247/248
(タイトクロップ)より明確に悪化した(詳細は上記報告書「原著本文の再精読で見つけた2つの差分」
節)。以後は`use_context_window=False`(タイトクロップ、247/248相当)を既定とする。

## 正規化スコープとバンドパスフィルタの選択制(2026-07-29)

`norm_scope`(`"trial"`|`"beat"`)で、RCG/ECGの正規化統計量(mean/std)をトライアル全体
から計算するか、拍セグメント自身から計算するかを選べる。`apply_bandpass=True`で、
RCG正規化前に心拍帯[1,25]Hzバンドパスフィルタをかける(`mmecg_sst_cache.py`のdocstring
参照。生の広帯域信号は体動由来と推測される高周波ノイズに正規化統計が支配されており、
10秒窓ごとのstdが最大9.3倍ばらつく問題を実測、バンドパス後は3.4倍まで縮小)。
**既定は`apply_bandpass=False`。** 245-249はこのオプション導入前に学習されており
`config.yaml`に該当キーが無いため、既定をTrueにすると`evaluate.py`でそれらのrunを
再評価した際に学習時と異なる設定でtest setが構築される(train/eval不整合)。253以降は
configに明示的に`apply_bandpass: true`を書く。253/254で検証した結果は248(バンドパス
なし)を下回った(悪化)ため、実際にも既定はFalseのままが望ましい。

`norm_scope="beat"`は、計算コスト上の制約(単一拍長の生信号からSSTを都度計算すると
50ch×1拍あたり約2.4秒かかり、9000拍規模では約6時間となり非現実的)から、**SST自体は
`norm_scope="trial"`と同じくトライアル全体で計算・キャッシュしたうえで、拍にクロップした
"後"にそのセグメント自身の統計量で再正規化する**という妥協実装になっている。真に生信号
段階から単一拍だけで計算し直す場合と厳密には異なる点に注意。
"""
from __future__ import annotations

import random
from collections import OrderedDict, deque
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset, Sampler

from dog_radar_vitals.data.mmecg import load_trial
from dog_radar_vitals.data.mmecg_ppi import (
    MAX_PPI_SEC,
    MIN_PPI_SEC,
    detect_consensus_beat_times,
    local_ppi_sec,
    sliding_ppi_estimates,
)
from dog_radar_vitals.data.mmecg_sst_cache import TIME_DECIMATION, get_or_compute_sst

T_FIXED_SST = 64  # タイトクロップ(既定)での時間分解能。文脈窓(使用時)はT_FIXED_SST_CONTEXTを使う。
T_FIXED_SST_CONTEXT = 128  # use_context_window=True時の時間分解能(原著Input SST形状に近づけた値)。
T_FIXED_ECG = 200
SST_FS = 200 / TIME_DECIMATION  # SSTキャッシュの実効サンプリングレート(decimation後)
CONTEXT_SEC = 4.0  # use_context_window=True時のSST入力窓の長さ(秒)。

NormScope = Literal["trial", "beat"]

Bounds = tuple[int, int, int, int]  # (sst_start, sst_end, ecg_start, ecg_end)


def _resize_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    """時間軸(最終軸)を線形補間で`target_len`にリサイズする。x: (..., T)。"""
    src_len = x.shape[-1]
    if src_len == target_len:
        return x
    src_idx = np.linspace(0, 1, src_len)
    tgt_idx = np.linspace(0, 1, target_len)
    flat = x.reshape(-1, src_len)
    resized = np.stack([np.interp(tgt_idx, src_idx, row) for row in flat], axis=0)
    return resized.reshape(*x.shape[:-1], target_len)


def _load_trial_context(
    raw_root: Path, trial_id: int, apply_bandpass: bool = False, ecg_norm_scope: NormScope = "trial"
) -> tuple[np.ndarray, np.ndarray, int]:
    """トライアルのSST(50, F, T_decim)・ECG(T,)・生fsを返す(いずれも遅延利用の元データ)。

    `ecg_norm_scope="trial"`: ECGはこの時点でトライアル全体のz-scoreに正規化される。
    `ecg_norm_scope="beat"`: ECGは正規化せず(NaN処理のみ)返す。正規化は`_segment_from_bounds`で
    拍セグメントを切り出した後、そのセグメント自身の統計量で行う。
    """
    sst, _ = get_or_compute_sst(raw_root, trial_id, apply_bandpass=apply_bandpass)
    rec = load_trial(raw_root, trial_id)
    ecg = rec.ecg.astype(np.float32)
    if ecg_norm_scope == "trial":
        ecg = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
    return sst, ecg, rec.fs


def _bounds_from_boundaries(boundaries: np.ndarray, ecg: np.ndarray, fs: int) -> list[Bounds]:
    """昇順の境界時刻列(秒)から、連続する区間ごとの(sst_start, sst_end, ecg_start, ecg_end)を作る。"""
    bounds: list[Bounds] = []
    for t, t_end in zip(boundaries[:-1], boundaries[1:]):
        if t_end - t < 0.2:  # 短すぎる断片はスキップ
            continue
        sst_start, sst_end = int(t * SST_FS), int(t_end * SST_FS)
        ecg_start, ecg_end = int(t * fs), min(int(t_end * fs), len(ecg))
        if sst_end - sst_start >= 2 and ecg_end - ecg_start >= 2 and not np.isnan(ecg[ecg_start:ecg_end]).any():
            bounds.append((sst_start, sst_end, ecg_start, ecg_end))
    return bounds


def _compute_bounds_legacy(sst: np.ndarray, ecg: np.ndarray, fs: int) -> list[Bounds]:
    """従来のPPIウォーク方式(局所PPI推定値の分だけ前進を繰り返す、開ループで誤差が蓄積しうる)。

    `detect_consensus_beat_times`が候補ピーク不足で失敗した場合のフォールバックとしてのみ使う。
    """
    estimates = sliding_ppi_estimates(sst, SST_FS)
    total_sec = sst.shape[-1] / SST_FS
    boundaries = [0.0]
    t = 0.0
    while t < total_sec:
        ppi = local_ppi_sec(estimates, t)
        t_end = min(t + ppi, total_sec)
        if t_end - t < 0.2:
            break
        boundaries.append(t_end)
        t = t_end
    return _bounds_from_boundaries(np.array(boundaries), ecg, fs)


def _compute_bounds(sst: np.ndarray, ecg: np.ndarray, fs: int) -> list[Bounds]:
    """拍タイミングのコンセンサス検出に基づき、連続する1心拍分の境界列を返す。"""
    total_sec = sst.shape[-1] / SST_FS
    beat_times = detect_consensus_beat_times(sst, SST_FS)
    if len(beat_times) < 3:  # コンセンサス検出が実質的に失敗(低SNRトライアル等)
        return _compute_bounds_legacy(sst, ecg, fs)

    midpoints = (beat_times[:-1] + beat_times[1:]) / 2
    boundaries = np.concatenate([[0.0], midpoints, [total_sec]])
    bounds = _bounds_from_boundaries(boundaries, ecg, fs)
    return [b for b in bounds if MIN_PPI_SEC * fs <= (b[3] - b[2]) <= MAX_PPI_SEC * 1.5 * fs]


def _segment_from_bounds(
    sst: np.ndarray,
    ecg: np.ndarray,
    bounds: Bounds,
    use_context_window: bool = False,
    norm_scope: NormScope = "trial",
    t_fixed_sst: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """1心拍分の境界からSSTセグメント(float16)・ECGセグメント(float32)を切り出す。

    `use_context_window=True`ならSST入力を拍中心のCONTEXT_SEC秒窓に、Falseなら
    タイトな拍境界にする(既定、247/248相当。理由はモジュールdocstring参照)。
    `norm_scope="beat"`なら、切り出し後にセグメント自身の統計量で再正規化する
    (SST: スケールのみ、ECG: mean+stdの標準化。モジュールdocstring「正規化スコープと
    バンドパスフィルタの選択制」参照)。

    `t_fixed_sst`(タイトクロップ時のみ有効。2026-08-01追加): `T_FIXED_SST`(既定64)を
    上書きしてSST時間分解能を変える。単一拍の生crop長は`SST_FS`(=100Hz)で概ね60〜100
    フレーム程度しかないため、64への線形補間リサイズは実質的な情報量をほぼ保っているが、
    backbone側の3段downsampling(各stride2、64→32→16→8)によりencoder出力の時間分解能が
    最終的に8フレーム(1拍あたり約87ms/フレーム)まで潰れてしまう。これはR波位置を
    ms単位で区別するには粗すぎる可能性がある(`reports/mmecg_comparison/
    prior_work_accuracy_comparison.md`「τ・Anchor補助タスクの公式実装への忠実化」節の
    結論: タイミング情報の学習が一貫して失敗する原因は入力側の時間分解能不足の可能性)。
    `t_fixed_sst`を128に上げ、backboneのencoder_out_timeを16に対応させることで、
    同じ3段downsampling比率のまま出力時間分解能を2倍(約43ms/フレーム)にできる。
    """
    sst_start, sst_end, ecg_start, ecg_end = bounds
    total_sst_len = sst.shape[-1]

    if use_context_window:
        center = (sst_start + sst_end) / 2.0
        half_window = CONTEXT_SEC * SST_FS / 2.0
        crop_start = int(round(max(center - half_window, 0)))
        crop_end = int(round(min(center + half_window, total_sst_len)))
        t_fixed_sst = T_FIXED_SST_CONTEXT
    else:
        crop_start, crop_end = sst_start, sst_end
        if t_fixed_sst is None:
            t_fixed_sst = T_FIXED_SST

    seg = sst[:, :, crop_start:crop_end].astype(np.float32)
    sst_seg = _resize_time_axis(seg, t_fixed_sst)
    ecg_seg = resample(ecg[ecg_start:ecg_end], T_FIXED_ECG).astype(np.float32)

    if norm_scope == "beat":
        sst_seg = sst_seg / (sst_seg.std() + 1e-8)  # 非負性を保つためscaleのみ(mean centeringしない)
        ecg_seg = (ecg_seg - ecg_seg.mean()) / (ecg_seg.std() + 1e-8)

    return sst_seg.astype(np.float16), ecg_seg


def estimate_cycle_bounds(raw_root: Path, trial_id: int, apply_bandpass: bool = False) -> list[Bounds]:
    sst, ecg, fs = _load_trial_context(raw_root, trial_id, apply_bandpass=apply_bandpass)
    return _compute_bounds(sst, ecg, fs)


def iter_single_cycles_with_bounds(
    raw_root: Path,
    trial_id: int,
    apply_bandpass: bool = False,
    norm_scope: NormScope = "trial",
    use_context_window: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """(SSTセグメント, ECGセグメント, ecg_start_idx, ecg_end_idx)のリストを返す。

    `ecg_start_idx`/`ecg_end_idx`(200Hz、生録音内のサンプル番号)は、
    `mmecg_radarode_longterm_dataset.py`が推定拍を元の時間軸に貼り戻す際に使う。
    """
    sst, ecg, fs = _load_trial_context(raw_root, trial_id, apply_bandpass=apply_bandpass, ecg_norm_scope=norm_scope)
    bounds_list = _compute_bounds(sst, ecg, fs)
    return [
        (*_segment_from_bounds(sst, ecg, bounds, use_context_window, norm_scope), bounds[2], bounds[3])
        for bounds in bounds_list
    ]


def iter_single_cycles(raw_root: Path, trial_id: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(SSTセグメント(50, T_FIXED_SST, F), ECGセグメント(T_FIXED_ECG,))のリストを返す。"""
    return [(sst, ecg) for sst, ecg, _, _ in iter_single_cycles_with_bounds(raw_root, trial_id)]


class _TrialSSTCache:
    """トライアルID -> SST全体(float16)、固定サイズLRU。"""

    def __init__(self, raw_root: Path, cache_size: int, apply_bandpass: bool = False) -> None:
        self._raw_root = raw_root
        self._cache_size = cache_size
        self._apply_bandpass = apply_bandpass
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, trial_id: int) -> np.ndarray:
        if trial_id in self._cache:
            self._cache.move_to_end(trial_id)
            return self._cache[trial_id]

        sst, _ = get_or_compute_sst(self._raw_root, trial_id, apply_bandpass=self._apply_bandpass)
        sst16 = sst.astype(np.float16)
        self._cache[trial_id] = sst16
        self._cache.move_to_end(trial_id)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return sst16


class TrialInterleavedSampler(Sampler[int]):
    """LRUキャッシュの局所性を保ちつつ、複数トライアルを束ねてシャッフルするSampler。

    完全ランダムシャッフルだとtrial単位でSST全体をロードし直すLRUキャッシュが毎サンプルで
    ミスしうる。`pool_size`個のトライアルだけを同時にアクティブにし、そのプール内で
    インデックスをランダムに取り出すことで、局所性(=キャッシュ命中率)を保ったまま
    近似的なシャッフルを実現する(TensorFlowの`interleave`に着想)。`pool_size`は
    `MMECGSingleCycleDataset`の`cache_size`と揃えること。
    """

    def __init__(self, trial_of_index: list[int], pool_size: int = 4, shuffle: bool = True, seed: int = 0) -> None:
        self.pool_size = pool_size
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._groups: dict[int, list[int]] = {}
        for i, t in enumerate(trial_of_index):
            self._groups.setdefault(t, []).append(i)

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        trial_order = list(self._groups.keys())
        if self.shuffle:
            rng.shuffle(trial_order)
        pending = deque(trial_order)
        pool: list[deque[int]] = []

        def refill() -> None:
            while pending and len(pool) < self.pool_size:
                t = pending.popleft()
                idxs = list(self._groups[t])
                if self.shuffle:
                    rng.shuffle(idxs)
                pool.append(deque(idxs))

        refill()
        while pool:
            i = rng.randrange(len(pool)) if self.shuffle else 0
            dq = pool[i]
            yield dq.popleft()
            if not dq:
                pool.pop(i)
                refill()

    def __len__(self) -> int:
        return sum(len(v) for v in self._groups.values())


class MMECGSingleCycleDataset(Dataset):
    """1心拍単位のSST->ECGペアを遅延ロードするDataset。

    `norm_scope`: `"trial"`(既定、247/248相当)はRCG/ECGともトライアル全体の統計量で
    正規化する。`"beat"`は拍セグメント自身の統計量で正規化する(モジュールdocstring参照)。
    `apply_bandpass`: RCG正規化前に心拍帯[1,25]Hzバンドパスフィルタをかけるか(既定True)。
    `use_context_window`: SST入力を拍中心の4秒窓にするか(既定False、悪化が確認済みのため)。
    `t_fixed_sst`: タイトクロップ時のSST時間分解能(既定None=`T_FIXED_SST`=64)。
    `_segment_from_bounds`のdocstring参照。
    """

    def __init__(
        self,
        raw_root: Path,
        trial_ids: list[int],
        cache_size: int = 4,
        norm_scope: NormScope = "trial",
        apply_bandpass: bool = False,
        use_context_window: bool = False,
        t_fixed_sst: int | None = None,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.norm_scope = norm_scope
        self.use_context_window = use_context_window
        self.t_fixed_sst = t_fixed_sst
        self._index: list[tuple[int, Bounds]] = []
        self._ecg_by_trial: dict[int, np.ndarray] = {}
        for trial_id in trial_ids:
            sst, ecg, fs = _load_trial_context(
                self.raw_root, trial_id, apply_bandpass=apply_bandpass, ecg_norm_scope=norm_scope
            )
            bounds_list = _compute_bounds(sst, ecg, fs)
            self._ecg_by_trial[trial_id] = ecg
            self._index.extend((trial_id, bounds) for bounds in bounds_list)
            del sst  # トライアル切替時に前トライアルのSSTを確実に解放する

        self.trial_of_index = [trial_id for trial_id, _ in self._index]
        self._sst_cache = _TrialSSTCache(self.raw_root, cache_size, apply_bandpass=apply_bandpass)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trial_id, bounds = self._index[idx]
        sst = self._sst_cache.get(trial_id)
        ecg = self._ecg_by_trial[trial_id]
        sst_seg, ecg_seg = _segment_from_bounds(
            sst, ecg, bounds, self.use_context_window, self.norm_scope, t_fixed_sst=self.t_fixed_sst
        )
        return torch.from_numpy(sst_seg.astype(np.float32)), torch.from_numpy(ecg_seg)
