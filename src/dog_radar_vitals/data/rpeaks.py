"""ECG波形からR波（心拍の基準点）を検出し、RR Intervalを計算する。

RR Interval予測は、密な波形再構成（`ecg_cnn1d.py`）とは異なる定式化
（疎なイベント検出）になりやすいという仮説を検証するため、Schellenbergerの
ヒトECGデータからR波の真値を作る。

検出法は単純な閾値+不応期方式（z-score化したECGでheight>2、最小間隔300ms）。
臨床用ECG（SNRが高い）ではこれで概ね十分だが、Pan-Tompkinsのような正式なQRS検出器
ではない点に注意。
"""
from __future__ import annotations

import bisect

import numpy as np
from scipy.signal import find_peaks

MIN_RR_SEC = 0.3  # 生理的に妥当な最小RR間隔（=最大200bpm相当）
PEAK_HEIGHT_ZSCORE = 2.0


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """ECG波形（生値）からR波のサンプルインデックスを検出する。

    2026-08-06追記: この方式はT波を第二のR波候補として検出したり、逆に真のR波を
    見逃したりすることがユーザ指摘・実データ(MMECGデータセット)で確認された
    (`dog-radar-vitals/reports/progress_report_2026-08-05_presentation.md`付録参照)。
    恒久対応としては`detect_r_peaks_neurokit`を使うこと。この関数は過去の
    再現性のために残す。
    """
    z = (ecg - np.nanmean(ecg)) / (np.nanstd(ecg) + 1e-8)
    peaks, _ = find_peaks(z, height=PEAK_HEIGHT_ZSCORE, distance=int(MIN_RR_SEC * fs))
    return peaks


def detect_r_peaks_neurokit(ecg: np.ndarray, fs: int) -> np.ndarray:
    """外部の検証済みQRS検出アルゴリズム(NeuroKit2)によるR波検出(恒久対応)。

    detect_r_peaksとの違い・採用理由は`radarODE-MTL/scripts/prepare_mmecg_dataset.py`の
    同名関数のdocstringに詳しい(このリポジトリとradarODE-MTL双方で同じ問題が
    確認されたため、同じ対応を独立に用意している)。MMECG全91ファイルでの検証では
    R-T混同を疑わせる短い(<350ms)拍間隔が0/19583件だった。

    対応する先行研究: Makowski et al., 2021, "NeuroKit2: A Python toolbox for
    neurophysiological signal processing," Behavior Research Methods。
    """
    import neurokit2 as nk

    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)


def rr_intervals_ms(peak_indices: np.ndarray, fs: int) -> np.ndarray:
    """隣接するR波間の時間間隔[ms]を返す（長さ=len(peak_indices)-1）。"""
    return np.diff(peak_indices) / fs * 1000.0


def build_peak_heatmap(peak_indices: np.ndarray, length: int, fs: int, sigma_ms: float = 10.0) -> np.ndarray:
    """R波位置に立てたガウシアンを重ね合わせた1次元heatmap（値域[0,1]）を作る。

    密な波形そのものではなく「R波がいつ起きたか」だけを回帰対象にすることで、
    QRSの振幅・形状を無視してタイミング検出に特化したターゲット表現になる。
    """
    heatmap = np.zeros(length, dtype=np.float32)
    sigma_samples = sigma_ms / 1000.0 * fs
    window = int(sigma_samples * 4)
    t = np.arange(-window, window + 1)
    gaussian = np.exp(-0.5 * (t / sigma_samples) ** 2)

    for idx in peak_indices:
        start = max(0, idx - window)
        end = min(length, idx + window + 1)
        g_start = start - (idx - window)
        g_end = g_start + (end - start)
        heatmap[start:end] = np.maximum(heatmap[start:end], gaussian[g_start:g_end])

    return heatmap


def extract_peaks_from_heatmap(heatmap: np.ndarray, fs: int, height: float = 0.3) -> np.ndarray:
    """予測heatmapからピーク位置を再抽出する（build_peak_heatmapの逆操作に相当）。"""
    peaks, _ = find_peaks(heatmap, height=height, distance=int(MIN_RR_SEC * fs))
    return peaks


def extract_peaks_paired_dedup(
    heatmap: np.ndarray,
    fs: int,
    height: float = 0.10,
    pair_merge_sec: float = 0.45,
    refractory_sec: float = 0.15,
) -> np.ndarray:
    """R波とT波の混同（隣接する二山）を緩和した、応急対応版のピーク抽出。

    背景（2026-08-06、ユーザ指摘・実データで確認済み）: このプロジェクトのR波検出
    （`detect_r_peaks`、z-score閾値+find_peaks）は、教師heatmap自体にT波を第二の
    R波候補として含めてしまうことがある。学習済みモデルの予測heatmapにも、真のR波
    検出直後に振幅の小さい第二の山が一貫して現れる（進捗報告書付録の図で確認）。
    恒久対応（外部の検証済みR波検出器で正解をつけ直し、再学習する）とは別に、
    「既に学習済みのモデルの後処理だけ」でこの症状を緩和する応急対応として、
    (1) 閾値を大きく下げて微弱な検出も候補に含め、(2) 真のRR間隔より明らかに短い
    間隔で隣接する候補どうしは前者（時間的に早い方=R波である可能性が高い方）だけを
    残す、という2段の後処理を行う。

    Parameters
    ----------
    height: 候補ピークの検出しきい値。extract_peaks_from_heatmapの0.3から大きく
        下げる（ユーザ指示: heatmap値で0.10かそれより低いくらい）。
    pair_merge_sec: これより短い間隔で隣接する2候補はR-T等のペアとみなし、前者を
        残す。真のRR間隔の生理的な下限（高強度運動後でも概ね300ms=200bpm相当）より
        明確に短くする必要はない（ユーザ指示: 「正確なRR間隔の最小値より短ければ
        何でもよい」）が、典型的なR-T間隔（QT間隔、概ね250〜450ms）は覆う必要がある
        ため、既定値は0.45秒とする。
    refractory_sec: find_peaks自体の不応期。真のQRS自体の不応期（150〜200ms程度）を
        大きく下回らない値にする。
    """
    candidates, _ = find_peaks(heatmap, height=height, distance=int(refractory_sec * fs))
    if len(candidates) < 2:
        return candidates

    kept = [candidates[0]]
    for c in candidates[1:]:
        gap_sec = (c - kept[-1]) / fs
        if gap_sec < pair_merge_sec:
            continue  # 直前に残した候補とのペア(R-T等)とみなし、後発は採用しない
        kept.append(c)
    return np.array(kept, dtype=candidates.dtype)


def extract_peaks_rhythmic(
    heatmap: np.ndarray,
    fs: int,
    candidate_height: float = 0.05,
    refractory_sec: float = 0.15,
    min_rr_sec: float = 0.3,
    max_rr_sec: float = 2.0,
    interval_tolerance_frac: float = 0.25,
    interval_tolerance_floor_sec: float = 0.05,
) -> np.ndarray:
    """予測heatmapの極大点から、間隔の規則性を根拠にR波の系列を選ぶ後処理（2026-08-06）。

    背景: `extract_peaks_paired_dedup`はR-T混同の応急対応として有効だが、モデル自体は
    T波以外にもさまざまな理由でノイズ状の疑似極大点を出しうる。ユーザ指摘: 学習の質では
    なく、heatmapから「どの極大点を採用するか」を選ぶアルゴリズムの質で検出率・RR
    Interval精度が改善するはず。特に、連続する極大点を2つずつのペアと見て、間隔が
    ほぼ一定で連続するペアの並びを採用すれば、(1) 近くに単発で出る雑音状の極大点は
    「一定間隔の並び」を構成できないため自然に棄却でき、(2) どのペアの並びも一定間隔に
    ならない区間があれば、真のR波の見逃し（or 閾値が低すぎる）を示唆する診断にもなる。

    アルゴリズム: 心拍のリズム（RR間隔がすぐ前の拍と近い値であること）を根拠にした、
    動的計画法によるビート系列選択（音楽情報処理のビートトラッキング手法、
    Ellis, 2007, "Beat Tracking by Dynamic Programming"と同型の定式化）。
    低いしきい値で拾った極大点候補それぞれについて、「直前の候補との間隔が、その候補が
    連なる系列の直前区間の間隔とどれだけ近いか」に応じたスコアを累積し、最終的に
    スコア最大の系列を1本だけ選ぶ。RR間隔が急に半分・倍になるような無関係な極大点
    （T波・体動由来のノイズ等）は、系列を継続する動機（スコア）がないため自然に外れる。
    HRの緩やかなドリフト（安静→運動等）は「直前区間との差」だけを見ているため許容される。

    Parameters
    ----------
    candidate_height: 極大点候補のしきい値。ノイズも含めて広く拾ってよい
        （どれを採用するかは間隔の規則性で選別するため、閾値自体は低くてよい）。
    min_rr_sec, max_rr_sec: 許容するRR間隔の範囲（デフォルトは生理的に妥当な
        30〜200bpm相当）。
    interval_tolerance_frac, interval_tolerance_floor_sec: 「一定間隔」とみなす許容誤差
        （直前区間の何割まで揺れを許すか、その下限[秒]）。この範囲を外れる遷移は
        系列として認めない（＝規則的な並びを構成できない候補は自然に脱落する）。
    """
    candidates, props = find_peaks(heatmap, height=candidate_height, distance=int(refractory_sec * fs))
    heights = props["peak_heights"]
    n = len(candidates)
    if n < 2:
        return candidates

    min_rr = min_rr_sec * fs
    max_rr = max_rr_sec * fs

    score = heights.astype(np.float64).copy()
    prev = np.full(n, -1, dtype=np.int64)
    prev_interval = np.full(n, -1.0)

    for i in range(n):
        lo = bisect.bisect_left(candidates, candidates[i] - max_rr)
        hi = bisect.bisect_right(candidates, candidates[i] - min_rr)
        for j in range(lo, hi):
            gap = candidates[i] - candidates[j]
            if prev_interval[j] < 0:
                consistency = 0.0
            else:
                tol = max(interval_tolerance_frac * prev_interval[j], interval_tolerance_floor_sec * fs)
                if abs(gap - prev_interval[j]) > tol:
                    continue  # 直前区間と間隔が違いすぎる = 規則的な並びとして認めない
                consistency = -abs(gap - prev_interval[j]) / fs
            cand_score = score[j] + heights[i] + consistency
            if cand_score > score[i]:
                score[i] = cand_score
                prev[i] = j
                prev_interval[i] = gap

    chain = []
    cur = int(np.argmax(score))
    while cur != -1:
        chain.append(cur)
        cur = int(prev[cur])
    chain.reverse()
    return candidates[np.array(chain, dtype=np.int64)]


def extract_peaks_adaptive_searchback(
    heatmap: np.ndarray,
    fs: int,
    candidate_height: float = 0.05,
    refractory_sec: float = 0.15,
    init_rr_sec: float = 0.8,
    missed_limit_frac: float = 1.66,
    searchback_threshold_frac: float = 0.5,
) -> np.ndarray:
    """Pan-Tompkinsの二重しきい値(signal/noise)+searchback方式をheatmap出力に適用した後処理。

    背景（2026-08-06、ユーザ指摘）: 固定しきい値による極大点選択は「時間成分を持たない
    線形な決定境界」に過ぎず、心拍のような周期信号の検出には本質的に不利。ECGのQRS検出
    分野では40年以上前から、直近の心拍間隔（RR間隔）の履歴に応じてしきい値を動的に
    調整し、さらに「予想される次拍のタイミングを大きく過ぎても検出できていない」場合は
    一度下げたしきい値で過去に遡って候補を探し直す（searchback）方式が標準になっている。

    対応する先行研究: Pan, J. and Tompkins, W. J., "A Real-Time QRS Detection Algorithm,"
    IEEE Transactions on Biomedical Engineering, 1985（Pan-Tompkinsアルゴリズム）。

    Parameters
    ----------
    candidate_height: 候補ピーク自体の下限（ノイズ除去のためこの値未満は最初から捨てる）。
    missed_limit_frac: 直近のRR間隔平均のこの倍率を超えて次のRが見つからない場合、
        見逃しとみなしてsearchbackを行う（原著は1.66）。
    searchback_threshold_frac: searchback時に使う、通常しきい値に対する緩和後の
        しきい値の比率（原著は0.5）。
    """
    candidates, props = find_peaks(heatmap, height=candidate_height, distance=int(refractory_sec * fs))
    heights = props["peak_heights"]
    n = len(candidates)
    if n < 2:
        return candidates

    init_n = min(8, n)
    spki = float(np.mean(heights[:init_n]))
    npki = float(np.min(heights[:init_n])) * 0.5
    rr_buffer: list[float] = [init_rr_sec * fs]

    accepted: list[int] = []
    last_r_idx = None

    for i in range(n):
        idx, h = int(candidates[i]), float(heights[i])
        thr1 = npki + 0.25 * (spki - npki)

        if last_r_idx is not None and rr_buffer:
            rr_avg = float(np.mean(rr_buffer))
            if idx - last_r_idx > missed_limit_frac * rr_avg:
                thr2 = searchback_threshold_frac * thr1
                window = [j for j in range(n) if last_r_idx < candidates[j] < idx and heights[j] > thr2]
                if window:
                    best_j = max(window, key=lambda j: heights[j])
                    accepted.append(int(candidates[best_j]))
                    spki = 0.25 * heights[best_j] + 0.75 * spki
                    rr_buffer.append(candidates[best_j] - last_r_idx)
                    if len(rr_buffer) > 8:
                        rr_buffer.pop(0)
                    last_r_idx = int(candidates[best_j])

        if h > thr1:
            accepted.append(idx)
            spki = 0.125 * h + 0.875 * spki
            if last_r_idx is not None:
                rr_buffer.append(idx - last_r_idx)
                if len(rr_buffer) > 8:
                    rr_buffer.pop(0)
            last_r_idx = idx
        else:
            npki = 0.125 * h + 0.875 * npki

    return np.array(sorted(set(accepted)), dtype=candidates.dtype)


def match_peaks(true_peaks: np.ndarray, pred_peaks: np.ndarray, fs: int, tolerance_ms: float = 50.0) -> dict:
    """真のR波と予測R波を、許容誤差内で1対1に対応付ける（貪欲法、時刻順）。

    2つの波形再構成アプローチ（密な波形回帰 vs heatmapキーポイント検出）を、検出したR波の
    タイミング精度・RR Intervalの精度という共通の物差しで比較するために使う。
    """
    tolerance_samples = tolerance_ms / 1000.0 * fs
    true_sorted = np.sort(true_peaks)
    pred_sorted = np.sort(pred_peaks)

    matched_true, matched_pred = [], []
    i, j = 0, 0
    while i < len(true_sorted) and j < len(pred_sorted):
        diff = pred_sorted[j] - true_sorted[i]
        if abs(diff) <= tolerance_samples:
            matched_true.append(true_sorted[i])
            matched_pred.append(pred_sorted[j])
            i += 1
            j += 1
        elif diff < 0:
            j += 1
        else:
            i += 1

    n_true, n_pred, n_matched = len(true_sorted), len(pred_sorted), len(matched_true)
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall = n_matched / n_true if n_true > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    timing_errors_ms = None
    if n_matched > 0:
        timing_errors_ms = (np.array(matched_pred) - np.array(matched_true)) / fs * 1000.0

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_true": np.array(matched_true),
        "matched_pred": np.array(matched_pred),
        "timing_errors_ms": timing_errors_ms,
    }


def matched_rr_interval_mae_ms(match_result: dict, fs: int) -> float | None:
    """マッチした真のR波の隣接ペアそれぞれについて、対応する予測RR IntervalとのMAE[ms]を返す。

    真の隣接R波2つが両方ともマッチできていた区間だけを比較対象にする（見逃し・過検出の
    影響を、RR Interval自体の精度評価からできるだけ切り離すため）。
    """
    matched_true = match_result["matched_true"]
    matched_pred = match_result["matched_pred"]
    if len(matched_true) < 2:
        return None

    true_rr = np.diff(matched_true) / fs * 1000.0
    pred_rr = np.diff(matched_pred) / fs * 1000.0
    return float(np.abs(true_rr - pred_rr).mean())
