"""RR間隔(タイムスタンプ付き)から個体内正規化した「覚醒度スコア」を計算する。

## 設計の要点(なぜこの形にしたか)

1. **入力インターフェースはECG特化にしない。**
   受け取るのは「R波が検出された時刻の系列」(`r_peak_times_s`, 単位は秒、単調増加)のみで、
   ECGの生波形やサンプリング周波数には一切依存しない。将来ミリ波レーダから推定したRR間隔
   (レーダ由来のR波相当タイミング)に入力を差し替える際、この関数より下流は変更不要にするため。
   ECGからR波時刻を取り出す部分(`rpeaks_from_ecg`)は補助関数として分離してあり、
   レーダ側のパイプラインはこの補助関数を経由しない。

2. **非重複窓(non-overlapping window)でHRを集計する。**
   `research/CLAUDE.md`の制約4「重複窓で自己相関やΔを測ってはいけない」に対応。
   窓を重ねると、隣接窓が同じ心拍を共有するために見かけ上の自己相関やなめらかな変化が
   生まれてしまう(過去に「42秒回復」という見かけの現象を作った失敗)。本実装は
   `windowing.py`の`nonoverlapping_windows`のみを使い、重複窓のオプション自体を提供しない。

3. **主指標はHR(bpm)、RMSSDは補助指標として分離した関数にする。**
   `research/CLAUDE.md`制約3「RMSSDは標本化ジッタに弱く上振れしやすい
   (RMSSD_meas ≈ sqrt(RMSSD_true^2 + 6*sigma_t^2))、HR単独の方が検出性能が高い」に対応。
   `arousal_score_from_rr`のデフォルト特徴量は窓内平均HRのみとし、RMSSDは
   `window_rmssd`という別関数として提供するに留め、覚醒度スコアの計算には使わない。

4. **正規化は被験者ごとのベースライン区間の平均・標準偏差によるz-score。**
   `research/CLAUDE.md`制約2「個体内正規化が必須(正規化なしAUC0.589→正規化ありAUC0.94)」に対応。
   `arousal_score_from_rr`は`baseline_mask`(どの窓がベースライン区間かを示すbool配列)を必須引数とし、
   ベースライン区間のHR平均・標準偏差で全窓のHRをz-score化した値を覚醒度スコアとする。

## 対応する先行研究・確立した知見

- 非重複窓でのHR/HRV計算という設計自体は、HRV解析の標準的な短時間窓解析
  (Task Force of ESC/NASPE, 1996, "Heart rate variability: standards of measurement...")
  に準拠しつつ、窓の重複を明示的に禁止する点で本プロジェクト固有の追加制約を課している。
- RMSSDの標本化ジッタ感受性は本プロジェクトの過去の実データ検証(`research/CLAUDE.md`記載、
  WESAD等での再現)で確立済みの知見であり、新たな文献根拠ではなく本プロジェクトの実測結果に基づく。
- 個体内z-score正規化によるHRV指標の被験者間差の吸収は、感情認識分野で広く使われる
  standard practice(例: Healey & Picard, 2005, "Detecting Stress During Real-World Driving
  Tasks Using Physiological Sensors"は個体ごとのベースライン正規化を採用している)に沿う。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import neurokit2 as nk
except ImportError:  # pragma: no cover - neurokit2は本プロジェクトの必須依存
    nk = None


@dataclass
class ArousalResult:
    """非重複窓ごとの覚醒度スコアと中間量。"""

    window_start_s: np.ndarray  # (n_windows,) 各窓の開始時刻[s]
    window_center_s: np.ndarray  # (n_windows,) 各窓の中心時刻[s]
    hr_bpm: np.ndarray  # (n_windows,) 窓内平均HR[bpm]
    rmssd_ms: np.ndarray  # (n_windows,) 窓内RMSSD[ms]。補助指標であり覚醒度スコアには使わない
    arousal: np.ndarray  # (n_windows,) 個体内z-score正規化した覚醒度スコア(主指標=HR由来)
    n_beats: np.ndarray  # (n_windows,) 各窓に含まれる心拍数(RR間隔の個数+1相当)
    baseline_mean_hr: float  # 正規化に使ったベースラインHR平均[bpm]
    baseline_std_hr: float  # 正規化に使ったベースラインHR標準偏差[bpm]


def rpeaks_from_ecg(ecg: np.ndarray, fs: float) -> np.ndarray:
    """ECG波形からR波の時刻[s]を検出する(neurokit2使用、プロジェクト内の既存手法と統一)。

    dog_radar_vitals/data/mmecg_beatgraph_dataset.py と同じ
    nk.ecg_clean -> nk.ecg_peaks の組み合わせを使う。
    """
    if nk is None:
        raise ImportError("neurokit2が必要です(pyproject.tomlの依存に含まれています)")
    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    rpeak_indices = np.asarray(rpeaks_info["ECG_R_Peaks"], dtype=np.int64)
    return rpeak_indices.astype(np.float64) / fs


def rr_intervals_from_r_peak_times(r_peak_times_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R波時刻の系列からRR間隔[s]と、各RR間隔の代表時刻(区間の終端時刻)を返す。

    レーダ由来のR波相当タイミングに差し替える際も、この関数から下流は変更不要。
    """
    r_peak_times_s = np.asarray(r_peak_times_s, dtype=np.float64)
    if r_peak_times_s.ndim != 1 or len(r_peak_times_s) < 2:
        raise ValueError("r_peak_times_sは長さ2以上の1次元配列である必要があります")
    rr_s = np.diff(r_peak_times_s)
    rr_times_s = r_peak_times_s[1:]  # 各RR間隔をその終端(2拍目)の時刻に紐づける
    return rr_s, rr_times_s


def window_hr_bpm(rr_s: np.ndarray, rr_times_s: np.ndarray, window_start: float, window_end: float) -> tuple[float, int]:
    """[window_start, window_end)に含まれるRR間隔から窓内平均HR[bpm]を計算する。

    HR = 60 / mean(RR[s]) 。窓内に心拍が無い場合はnanを返す。
    """
    in_window = (rr_times_s >= window_start) & (rr_times_s < window_end)
    rr_in = rr_s[in_window]
    if len(rr_in) == 0:
        return float("nan"), 0
    mean_rr_s = float(np.mean(rr_in))
    if mean_rr_s <= 0:
        return float("nan"), int(len(rr_in))
    return 60.0 / mean_rr_s, int(len(rr_in))


def window_rmssd(rr_s: np.ndarray, rr_times_s: np.ndarray, window_start: float, window_end: float) -> float:
    """[window_start, window_end)内のRMSSD[ms]を計算する(補助指標。覚醒度スコアには使わない)。

    RMSSDは連続するRR間隔の差分に基づくため、標本化ジッタの影響を強く受ける
    (research/CLAUDE.md制約3)。主指標として採用しないが、比較のために計算・出力はしておく。
    """
    in_window = (rr_times_s >= window_start) & (rr_times_s < window_end)
    rr_in = rr_s[in_window]
    if len(rr_in) < 2:
        return float("nan")
    diffs_ms = np.diff(rr_in) * 1000.0
    return float(np.sqrt(np.mean(diffs_ms**2)))


def arousal_score_from_rr(
    r_peak_times_s: np.ndarray,
    baseline_mask_fn,
    window_sec: float = 10.0,
    session_start_s: float | None = None,
    session_end_s: float | None = None,
) -> ArousalResult:
    """RR間隔系列から非重複窓ごとの覚醒度スコア(個体内z-score化したHR)を計算する。

    Parameters
    ----------
    r_peak_times_s: R波(相当)時刻の系列[s]、単調増加。ECG由来でもレーダ由来でも可。
    baseline_mask_fn: 窓の中心時刻[s]を受け取り、その窓がベースライン区間(安静時)に
        属するかどうかのboolを返す関数。呼び出し側がデータの意味(video_id等)を知っているため、
        「どこがベースラインか」の判定はこの関数実装側の責務とし、本関数はそれを使うだけにする。
    window_sec: 非重複窓の長さ[s]。デフォルト10秒(タスク指定の例)。
    session_start_s, session_end_s: 窓を切る範囲。省略時はr_peak_times_sの範囲。

    Returns
    -------
    ArousalResult: 覚醒度スコア(主指標=HRのz-score)を含む窓ごとの結果。
    """
    rr_s, rr_times_s = rr_intervals_from_r_peak_times(r_peak_times_s)

    start = session_start_s if session_start_s is not None else float(r_peak_times_s[0])
    end = session_end_s if session_end_s is not None else float(r_peak_times_s[-1])

    from .windowing import nonoverlapping_windows

    window_starts = nonoverlapping_windows(start, end, window_sec)
    n = len(window_starts)

    hr = np.full(n, np.nan)
    rmssd = np.full(n, np.nan)
    n_beats = np.zeros(n, dtype=np.int64)
    centers = window_starts + window_sec / 2.0
    is_baseline = np.zeros(n, dtype=bool)

    for i, ws in enumerate(window_starts):
        we = ws + window_sec
        hr[i], n_beats[i] = window_hr_bpm(rr_s, rr_times_s, ws, we)
        rmssd[i] = window_rmssd(rr_s, rr_times_s, ws, we)
        is_baseline[i] = bool(baseline_mask_fn(centers[i]))

    baseline_hr = hr[is_baseline]
    baseline_hr = baseline_hr[~np.isnan(baseline_hr)]
    if len(baseline_hr) < 2:
        raise ValueError(
            "ベースライン区間に有効なHR推定窓が2未満です。baseline_mask_fnまたはwindow_secを確認してください。"
        )
    baseline_mean = float(np.mean(baseline_hr))
    baseline_std = float(np.std(baseline_hr, ddof=1))
    if baseline_std == 0.0:
        baseline_std = 1e-8  # 定数区間で分散ゼロになる病的ケースの回避

    arousal = (hr - baseline_mean) / baseline_std

    return ArousalResult(
        window_start_s=window_starts,
        window_center_s=centers,
        hr_bpm=hr,
        rmssd_ms=rmssd,
        arousal=arousal,
        n_beats=n_beats,
        baseline_mean_hr=baseline_mean,
        baseline_std_hr=baseline_std,
    )
