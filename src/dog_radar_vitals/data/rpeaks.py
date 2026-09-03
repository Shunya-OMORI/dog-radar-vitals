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


def detect_r_and_t_peaks_neurokit(ecg: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """NeuroKit2でR波とT波の両方を検出する（2026-08-06、ユーザ発案の派生案）。

    背景: heatmapモデルはR波の教師しか与えられていないのに，予測heatmapは1心拍に
    つき隣接する2つの山（R波＋T波）を出しがちだった．これは「T波を誤検出している」
    というより，レーダ反射信号にはR波・T波双方に対応する物理的な変化が実際に存在し，
    モデルがそれを拾ってしまうのが自然，という可能性を示唆する．ならば「T波を予測
    するな」と矛盾した教師信号を与え続けるより，**R波・T波の両方を正解として与え，
    どちらがRかの選別は後処理（既知のR→T間隔の規則性を使える）に任せる**方が，
    モデルにとって一貫した学習課題になるのではという発想．

    対応する先行研究: Makowski et al., 2021 (NeuroKit2)．`ecg_delineate`による
    QRS/T波の同時デリニエーションは，姿勢推定分野で複数の補助的なランドマークを
    同時に検出させてから幾何的制約で本命を絞り込む手法（例: 顔・手のランドマーク
    検出）と同型の発想．

    Returns
    -------
    (r_peaks, t_peaks): 両方ともサンプルインデックスのndarray．T波はR波と1対1に
        対応するとは限らない（デリニエーションに失敗した拍はnanとして除外される）。
    """
    import neurokit2 as nk

    cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    _, r_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    r_peaks = np.asarray(r_info["ECG_R_Peaks"], dtype=np.int64)

    _, waves_info = nk.ecg_delineate(cleaned, rpeaks=r_info, sampling_rate=fs, method="dwt")
    t_peaks_raw = np.asarray(waves_info["ECG_T_Peaks"], dtype=np.float64)
    t_peaks = t_peaks_raw[~np.isnan(t_peaks_raw)].astype(np.int64)
    return r_peaks, t_peaks


def rr_intervals_ms(peak_indices: np.ndarray, fs: int) -> np.ndarray:
    """隣接するR波間の時間間隔[ms]を返す（長さ=len(peak_indices)-1）。"""
    return np.diff(peak_indices) / fs * 1000.0


def build_peak_heatmap(peak_indices: np.ndarray, length: int, fs: int, sigma_ms: float = 10.0) -> np.ndarray:
    """R波位置に立てたガウシアンを重ね合わせた1次元heatmap（値域[0,1]）を作る。

    密な波形そのものではなく「R波がいつ起きたか」だけを回帰対象にすることで、
    QRSの振幅・形状を無視してタイミング検出に特化したターゲット表現になる。
    """
    heatmap = np.zeros(length, dtype=np.float32)

    # sigma_ms <= 0 は「幅を持たない教師」= R波位置だけを1にする 0/1 スパイク列。
    # radarODE 系の公式実装が配布している anchor はこの形である
    # (CFT-RFcardi の data_anchor_*.npy を実測: 値は {0,1} のみ、4秒窓に4〜7点)。
    # σ→0 の極限として比較できるようにするための分岐で、既存の σ>0 の挙動は変えない。
    if sigma_ms <= 0:
        idx = np.asarray(peak_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < length)]
        heatmap[idx] = 1.0
        return heatmap

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


def build_peak_box(peak_indices: np.ndarray, length: int, fs: int, half_width_ms: float = 150.0) -> np.ndarray:
    """R波位置を中心とする矩形(0/1)マスクを作る（QRS検出のU-Net系文献で一般的なターゲット表現）。

    ガウシアン(build_peak_heatmap)・0/1スパイク(build_peak_heatmap, sigma_ms<=0)とは異なる
    第3の選択肢。半値幅そのものを1で埋めるため、「ピーク位置」ではなく「区間」を教師にする。
    先行研究例: Sereda ら(U-Net+BiLSTM, arXiv:1912.09223)、QRS Detection via ECG segmentation
    (IEEE T-IM 2023, 10.1109/TIM.2023.3236316)。デフォルト半値幅150msはこれらの文献の
    典型値(±100〜200ms)の中央付近。
    """
    mask = np.zeros(length, dtype=np.float32)
    half_width = int(half_width_ms / 1000.0 * fs)
    for idx in peak_indices:
        start = max(0, idx - half_width)
        end = min(length, idx + half_width + 1)
        mask[start:end] = 1.0
    return mask


def extract_peaks_from_box(mask: np.ndarray, fs: int, height: float = 0.5) -> np.ndarray:
    """矩形マスク予測から、各連結成分(閾値以上の区間)の中心をR波位置として抽出する。"""
    above = mask >= height
    peaks = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            peaks.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return np.array(peaks, dtype=int)


def extract_peaks_dark_refine(heatmap: np.ndarray, fs: int, height: float = 0.3, smooth_sigma_samples: float = 2.0) -> np.ndarray:
    """DARK(Zhang ら, CVPR 2020, arXiv:1910.06278)のTaylor展開デコードを1次元に移植した後処理。

    学習済みheatmapをGaussianで平滑化してからfind_peaksで粗い位置を取り、各ピーク周りの
    対数heatmapを2次のTaylor展開で近似してサブサンプル単位の補正量Δ=-f'/f''を加える。
    モデル・学習は一切変更せず、既存チェックポイントの予測heatmapに対する後処理の差し替え
    のみ(extract_peaks_from_heatmapとの比較用)。
    """
    from scipy.ndimage import gaussian_filter1d

    smoothed = gaussian_filter1d(heatmap.astype(np.float64), sigma=smooth_sigma_samples)
    coarse_peaks, _ = find_peaks(smoothed, height=height, distance=int(MIN_RR_SEC * fs))

    log_hm = np.log(np.clip(smoothed, 1e-6, None))
    refined = []
    for p in coarse_peaks:
        if 1 <= p < len(log_hm) - 1:
            d1 = (log_hm[p + 1] - log_hm[p - 1]) / 2.0
            d2 = log_hm[p + 1] - 2 * log_hm[p] + log_hm[p - 1]
            delta = -d1 / d2 if abs(d2) > 1e-6 else 0.0
            delta = float(np.clip(delta, -1.0, 1.0))  # DARK論文通り±1サンプルに制限
            refined.append(p + delta)
        else:
            refined.append(float(p))
    return np.round(np.array(refined)).astype(int)


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
    threshold_frac: float = 0.25,
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
    threshold_frac: 通常時の受理しきい値 thr1 = NPKI + threshold_frac*(SPKI-NPKI) の
        係数（原著は0.25）。原著はQRS前処理後の信号でSN比が非常に高いことを前提にした
        値であり、SN分離が乏しい入力（heatmap出力等）では大きめの値が必要になりうる。
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
        thr1 = npki + threshold_frac * (spki - npki)

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


def refine_peaks_centroid(
    heatmap: np.ndarray,
    peaks: np.ndarray,
    fs: int,
    half_win_ms: float = 85.0,
    n_iter: int = 2,
) -> np.ndarray:
    """検出済みピーク位置を、heatmap近傍の重み付き重心(soft-argmax)でサブサンプル精度に補正する。

    heatmapはfs=200Hz(1サンプル=5ms)に離散化されており、find_peaksの整数argmaxには
    量子化誤差(±2.5ms)と、ノイズ起因のジッタが乗る。ガウシアン教師で学習したheatmapの
    山は正解位置の周りに質量を持つため、山全体の重心は単一の極大点よりも安定した
    位置推定量になる(2026-08-07、config274で matched timing error std 28.8→27.0ms、
    RR-MAE 12.00→8.29ms を確認)。

    対応する先行研究:
    - soft-argmax(重み付き重心)による連続座標復号: Sun et al., "Integral Human Pose
      Regression", ECCV 2018。
    - heatmap argmaxの量子化誤差をピーク近傍の分布形状で補正する枠組み:
      Zhang et al., "Distribution-Aware Coordinate Representation for Human Pose
      Estimation" (DARK pose), CVPR 2020。

    Parameters
    ----------
    half_win_ms: 重心をとる窓の半幅。σ=10-15msの教師に対し山の裾全体(±5σ超)を覆う
        85ms付近が最良で、75-110msに広いプラトーがある(test/2チェックポイントで確認)。
        250ms先のT波由来の副峰に届く幅(>150ms)まで広げると悪化する。
    n_iter: 重心位置に窓を再センタリングして繰り返す回数。2回でほぼ収束する。

    Returns
    -------
    連続値(float)のピーク位置。RR Interval計算にはそのまま使える。
    """
    pos = np.asarray(peaks, dtype=np.float64)
    hw = max(1, int(round(half_win_ms / 1000.0 * fs)))
    for _ in range(n_iter):
        centers = np.round(pos).astype(int)
        out = []
        for p in centers:
            lo, hi = max(0, p - hw), min(len(heatmap), p + hw + 1)
            seg = heatmap[lo:hi].astype(np.float64)
            w = np.maximum(seg - seg.min(), 0.0)
            if w.sum() <= 0:
                out.append(float(p))
                continue
            out.append(float(np.sum(np.arange(lo, hi) * w) / w.sum()))
        pos = np.array(out)
    return pos


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


def extract_peaks_viterbi(
    heatmap: np.ndarray,
    fs: int,
    candidate_height: float = 0.02,
    min_rr_sec: float = 0.30,
    max_rr_sec: float = 2.00,
    prior_center_sec: float | None = None,
    prior_sigma_sec: float = 0.25,
    prior_weight: float = 1.0,
) -> np.ndarray:
    """heatmap から拍の系列を動的計画法で最尤推定する復号器 (2026-08-25 追加)。

    しきい値方式との違い
    --------------------
    しきい値方式は各極大点を**独立に**「拍か否か」判定するので、確信度が下がった
    区間の拍がまるごと消える。11-fold LOSO の診断では precision 0.93〜1.00 に対し
    recall が被験者 9 で 0.13 まで落ち、MDR は 92.6% だった。一方、見つかった拍の
    間隔は正確 (RR間隔MAE は全被験者 8.25〜14.29 ms) で、位置精度ではなく
    **検出の取りこぼし**だけが崩れている。

    心拍は生理的に「必ず一定の範囲の間隔で続く」ので、拍を独立に判定するのは
    情報を捨てている。ここでは拍の系列全体のスコア

        S = sum_j [ log h(t_j) + prior_weight * log P(t_j - t_{j-1}) ]

    を最大化する系列を Viterbi 型の DP で求める。P は RR 間隔の事前分布で、
    生理的範囲 [min_rr_sec, max_rr_sec] の外は確率 0、内側は緩いガウシアン。

    Ellis のビート追跡 (024 で実装・評価済み) との違い
    -------------------------------------------------
    Ellis 型は「テンポ一定」を仮定して逸脱を罰するので、RR 間隔の変動そのものを
    潰す。実測でも F1 は 0.571→0.643 に上がったが RMSSD 比は 1.550→1.734 に悪化し、
    HRV を目的にする以上は採用できなかった。ここでは事前分布の幅
    (prior_sigma_sec 既定 0.25 s) を生理的変動より十分広く取り、
    **拍が続くことだけを制約して、間隔の変動は制約しない**。
    prior_weight=0 にすると間隔の事前分布を完全に外せる (連鎖の制約だけが残る)。

    イヌへの転用でこの設計が効く理由: イヌの心拍数は 60〜180 bpm と個体差・
    状況差が大きく、呼吸性洞性不整脈による拍ごとの変動もヒトより大きい。
    テンポ一定を仮定する手法はそこで破綻するが、この復号は生理的範囲の広さと
    事前分布の幅を変えるだけで種を移せる。

    対応する先行研究:
      - Viterbi, "Error bounds for convolutional codes and an asymptotically
        optimum decoding algorithm," IEEE Trans. Information Theory, 1967
      - Coast, Stern, Cano, Briller, "An approach to cardiac arrhythmia analysis
        using hidden Markov models," IEEE Trans. Biomedical Engineering, 1990
      - Ellis, "Beat tracking by dynamic programming," J. New Music Research, 2007

    Parameters
    ----------
    candidate_height: 候補にする極大点の下限。しきい値方式の height と違い、
        ここは「捨てすぎない」ためだけの値なので低く取る (既定 0.02)。
    prior_center_sec: RR 間隔事前分布の中心。None なら候補間隔の中央値から推定する。
    prior_weight: 事前分布項の重み。0 で間隔の事前分布を無効化。
    """
    heatmap = np.asarray(heatmap, dtype=float)
    cand, props = find_peaks(heatmap, height=candidate_height,
                             distance=max(1, int(min_rr_sec * fs)))
    if len(cand) < 2:
        return cand
    h = np.clip(props["peak_heights"], 1e-6, 1.0)
    t = cand.astype(float) / fs

    if prior_center_sec is None:
        d = np.diff(t)
        d = d[(d >= min_rr_sec) & (d <= max_rr_sec)]
        prior_center_sec = float(np.median(d)) if d.size else 0.8

    # スコアは Ellis (2007) と同じ「報酬の和」の形にする。heatmap 値 h は正の報酬なので
    # 拍を取るほど得になり、事前分布項 lp は負のペナルティなので生理的にありえない
    # 間隔をつなぐと損になる。対数尤度の和にすると log h < 0 で拍を取るほどスコアが
    # 下がり、連鎖しない系列が常に勝ってしまう。
    n = len(cand)
    score = h.copy()
    back = np.full(n, -1, dtype=int)
    for j in range(1, n):
        best, best_i = score[j], -1
        for i in range(j - 1, -1, -1):
            gap = t[j] - t[i]
            if gap > max_rr_sec:
                break
            if gap < min_rr_sec:
                continue
            lp = -0.5 * ((gap - prior_center_sec) / prior_sigma_sec) ** 2
            cand_score = score[i] + h[j] + prior_weight * lp
            if cand_score > best:
                best, best_i = cand_score, i
        score[j], back[j] = best, best_i

    j = int(np.argmax(score))
    path = []
    while j >= 0:
        path.append(cand[j])
        j = back[j]
    return np.array(path[::-1], dtype=cand.dtype)


def correct_rr_artifacts_kubios(peaks: np.ndarray, fs: int, iterative: bool = True) -> np.ndarray:
    """検出した拍の系列から、見逃し・余分な拍・位置ずれを補正する (2026-08-25 追加)。

    背景: 我々の RMSSD は正解の 1.584 倍に過大推定されており、その内訳は
    ジッタ由来が 0.128、**見逃し/誤検出由来が 0.456** だった。見逃した拍をまたぐ
    RR 間隔は約 2 倍になり、差分を取る RMSSD を直接壊す。単純な外れ値除去
    (`clean_rr_intervals_ms`) では、そこに拍を復元できないので誤差が残る。

    Lipponen & Tarvainen (2019) の補正はこれを型に分けて扱う。
      - 連続 RR 差 dRR の分布から求めた四分位偏差 x 5.2 を時変しきい値 Th とする
        (前後 90 拍の窓。正規分布を仮定して 99.95% を覆う係数)
      - |RR(i)/2 - medRR(i)| < 2*Th なら **見逃し** とみなし、R 波を 1 つ挿入する
      - |RR(i) + RR(i+1) - medRR(i)| < 2*Th なら **余分** とみなし、1 つ取り除く
      - dRR の符号パターン NPN/PNP は異所性拍、PN は long、NP は short として補間で直す
    HRV 解析の標準ソフト Kubios に採用されている手法で、NeuroKit2 に
    `signal_fixpeaks(method="Kubios")` として実装されている。

    対応する論文: Lipponen & Tarvainen, "A robust algorithm for heart rate variability
    time series artefact correction using novel beat classification,"
    Journal of Medical Engineering & Technology 43(3):173-181, 2019.
    https://www.tandfonline.com/doi/full/10.1080/03091902.2019.1640306

    Viterbi 復号 (`extract_peaks_viterbi`) との役割分担: Viterbi は「拍が続く」制約で
    **見逃しを減らす**が、その代償に拍を取りすぎて pNN50 と平均 IBI を壊すことが
    被験者 9 の試走で分かっている。こちらは取りすぎた拍を型として検出して外せるので、
    2 つは補い合う。
    """
    peaks = np.asarray(peaks)
    if peaks.size < 4:
        return peaks
    try:
        import neurokit2 as nk
        out = nk.signal_fixpeaks(peaks, sampling_rate=fs, iterative=iterative,
                                 method="Kubios", show=False)
        # neurokit2 のバージョンによって (artifacts, peaks_clean) か peaks_clean を返す
        corrected = out[1] if isinstance(out, tuple) else out
        corrected = np.asarray(corrected)
        if corrected.size == 0:
            return peaks
        return np.sort(corrected.astype(peaks.dtype))
    except Exception:
        # 補正に失敗したら黙って元の系列を返す(評価が止まらないようにする)
        return peaks


def rr_intervals_with_validity(
    peaks: np.ndarray,
    fs: int,
    min_rr_ms: float = 300.0,
    max_rr_ms: float = 1500.0,
    rel_dev: float = 0.2,
    median_window: int = 11,
    use_local_median: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """RR 間隔と、その各要素が信頼できるかのフラグを返す (2026-08-25 追加)。

    既存の `evaluate_anchor_hrv_metrics.clean_rr_intervals_ms` には 2 つの欠陥があった。

    1. docstring は "local median of its neighbors" と書いているが、実装は
       `np.median(rr_valid)` で **系列全体の中央値**を使っていた。心拍が変動する試行
       (運動後など) では正常な拍まで外れ値と判定されてしまう。
    2. 外れ値を除いた RR を **連結してから** `np.diff` を取っていた。
       例えば [800, 1600(見逃し), 900] から 1600 を除くと [800, 900] になり、
       本来は隣接していない 800 と 900 の差 100 ms を「隣接拍の変動」として数える。
       **RMSSD は差分の差分なので、この連結が直撃する。**

    ここでは除外せず、**マスクを返す**。RMSSD は「両隣とも有効な差分」だけで計算すればよい
    (`rmssd_from_masked_rr`)。これは HRV 解析でアーティファクト区間を除外する標準的な扱いで、
    Task Force (Circulation 93(5):1043-1065, 1996) も補間より除外を基本としている。

    局所中央値は前後 `median_window` 拍の移動中央値を使う。

    Returns
    -------
    rr : (N,) RR 間隔 [ms]
    valid : (N,) bool。生理的範囲内かつ局所中央値から rel_dev 以内なら True
    """
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 3:
        return np.array([]), np.array([], dtype=bool)
    rr = np.diff(peaks) / fs * 1000.0
    valid = (rr >= min_rr_ms) & (rr <= max_rr_ms)
    if valid.sum() < 3:
        return rr, valid

    if use_local_median:
        # 局所中央値: 有効な要素だけを使って、前後 median_window 拍の移動中央値を取る。
        # **既定を False にした理由 (2026-08-25、11-fold LOSO で切り分け済み)**:
        #   (a) legacy(全体中央値・連結あり)      RMSSD 16.94 ms / SDNN  8.15 ms
        #   (b) マスク方式・全体中央値            RMSSD 15.46 ms / SDNN  8.15 ms  ← 採用
        #   (c) マスク方式・局所中央値            RMSSD 15.40 ms / SDNN 19.94 ms  ← SDNN が壊れる
        # 連続した誤検出があると局所中央値自体がそれに引きずられ、外れ値を「正常」と
        # 判定してしまう。全体中央値は心拍が変わる試行に弱く、局所中央値は連続外れ値に弱い。
        # 我々のデータでは後者の害の方が大きかったので、既定は全体中央値にする。
        half = max(1, median_window // 2)
        local_med = np.empty_like(rr)
        for i in range(len(rr)):
            lo, hi = max(0, i - half), min(len(rr), i + half + 1)
            win = rr[lo:hi][valid[lo:hi]]
            local_med[i] = np.median(win) if win.size else np.median(rr[valid])
    else:
        local_med = np.full_like(rr, float(np.median(rr[valid])))
    valid &= np.abs(rr - local_med) <= rel_dev * local_med
    return rr, valid


def rmssd_from_masked_rr(rr: np.ndarray, valid: np.ndarray) -> float | None:
    """連続する 2 つの RR 間隔がどちらも有効なときだけ差分を取って RMSSD を計算する。

    除外区間をまたぐ差分を使わないので、見逃し・誤検出が RMSSD を膨らませない。
    """
    if rr.size < 3 or valid.sum() < 3:
        return None
    d = np.diff(rr)
    pair_valid = valid[:-1] & valid[1:]
    if pair_valid.sum() < 2:
        return None
    return float(np.sqrt(np.mean(d[pair_valid] ** 2)))


def sdnn_from_masked_rr(rr: np.ndarray, valid: np.ndarray) -> float | None:
    """SDNN は差分を取らないので、有効な RR だけの標準偏差でよい。"""
    if valid.sum() < 3:
        return None
    return float(np.std(rr[valid]))


def pnn50_from_masked_rr(rr: np.ndarray, valid: np.ndarray) -> float | None:
    """pNN50 も RMSSD と同じく、両隣が有効な差分だけを数える。"""
    if rr.size < 3 or valid.sum() < 3:
        return None
    d = np.abs(np.diff(rr))
    pair_valid = valid[:-1] & valid[1:]
    if pair_valid.sum() < 2:
        return None
    return float(np.mean(d[pair_valid] > 50.0) * 100.0)


def fill_missed_beats_median(
    peaks: np.ndarray,
    fs: int,
    median_window: int = 11,
    gap_ratio: float = 1.5,
    max_insert: int = 4,
    jitter_from_neighbors: bool = False,
) -> np.ndarray:
    """局所中央値を使って、見逃した拍を挿入する (2026-08-25 追加)。

    Kubios (Lipponen & Tarvainen, 2019) の補正は系列全体の四分位偏差からしきい値を作るが、
    ここは 2 点を変えている。

    1. **局所中央値を使う。** 前後 `median_window` 拍の移動中央値と比べて
       `gap_ratio` 倍を超える間隔を「見逃し」と判定する。心拍数が試行中に変わる
       (MMECG には運動後の条件がある) 場合、全体統計では正常な拍まで巻き込む。
    2. 挿入位置の与え方を選べる。当初は「等分割だと挿入区間の隣接差分が 0 になり
       RMSSD を過小評価するはず」と考えて、前後の実測 RR のばらつきに合わせて
       挿入位置を揺らす方式 (`jitter_from_neighbors=True`) を用意した。
       **しかし実測では逆だった。** 120 拍から 4 拍を抜いた合成系列で
         正解 RMSSD 50.76 ms / 等分割 50.01 ms / 揺らす方式 84.37 ms
       となり、揺らす方式は大きく過大評価した。挿入した拍の前後で差分が 2 回生じ、
       与えた揺らぎがそのまま RMSSD に乗るためである。**既定は等分割 (False) にした。**

    ただし、挿入した拍は測定値ではないので、HRV を厳密に測るなら
    **挿入せずに除外する** (`rr_intervals_with_validity` + `rmssd_from_masked_rr`) 方が正しい。
    Task Force (Circulation 93(5):1043-1065, 1996) も補間より除外を基本としている。
    この関数は「拍の系列そのものが欲しい」場面 (可視化、拍数の推定) のために置く。

    Returns
    -------
    挿入後の拍位置 (サンプル単位、昇順)
    """
    peaks = np.asarray(peaks, dtype=float)
    if len(peaks) < 4:
        return peaks
    rr = np.diff(peaks)
    half = max(1, median_window // 2)
    out = [peaks[0]]
    for i, gap in enumerate(rr):
        lo, hi = max(0, i - half), min(len(rr), i + half + 1)
        win = rr[lo:hi]
        med = float(np.median(win)) if win.size else float(np.median(rr))
        n_missing = int(round(gap / med)) - 1 if med > 0 else 0
        n_missing = max(0, min(n_missing, max_insert))
        if n_missing > 0 and gap > gap_ratio * med:
            if jitter_from_neighbors and win.size >= 3:
                # 実測で過大評価すると分かっている経路 (上の docstring 参照)。
                # 比較のために残してあるだけで、既定では使わない。
                sd = float(np.std(win))
                rng = np.random.RandomState(int(peaks[i]) % (2 ** 31))
                w = np.abs(rng.normal(med, sd, n_missing + 1))
                w = w / w.sum() * gap
            else:
                w = np.full(n_missing + 1, gap / (n_missing + 1))
            pos = peaks[i]
            for k in range(n_missing):
                pos += w[k]
                out.append(pos)
        out.append(peaks[i + 1])
    return np.asarray(sorted(out), dtype=peaks.dtype)


def normalize_heatmap(heatmap: np.ndarray, fs: int, kind: str = "rank") -> np.ndarray:
    """出力 heatmap を録画ごとに正規化してから、しきい値を掛けられるようにする (2026-08-26 追加)。

    背景（ユーザ提案）: MAD 正規化は入力の 50 チャネルにだけ掛けていた。しきい値 h は
    heatmap の生の値に対する固定値なので、被験者によって heatmap 全体の高さが違うと
    同じ h でも意味が変わる。実際、被験者ごとの最良 h は 0.05〜0.20 に散らばっており、
    「テストの被験者を見ずに動作点を決められない」ことが最大の実用上の課題だった。

    **効果（11分割 LOSO、入れ子の被験者分割で検証済み）**:

    | 正規化 | 他10名で h を選んだときの F1 | RMSSD 誤差 |
    |---|---|---|
    | none (従来)    | 0.818 ± 0.119 | 18.20 ms |
    | mad            | 0.857 ± 0.072 | 17.54 ms |
    | p99            | 0.867 ± 0.072 | 15.05 ms |
    | **rank (採用)** | **0.888 ± 0.057** | **14.26 ms** |
    | (参考) 入力を見ずに一定 RR を出すだけ | — | 41.34 ms |

    さらに、各被験者について「他 10 名で正規化と h の両方を選ぶ」入れ子の検証を行うと、
    **11 名全員で rank / h=0.90 が選ばれた**。選択が偶然でないことの証拠になっている。

    なぜ rank が効くか: 順位変換は値の分布の形を完全に潰し「上位何 % か」だけを残す。
    被験者ごとに山の高さが違っても「上位 10 % を拍の候補とみなす」という基準は共通に使える。
    心拍数がほぼ一定なら、拍が占める時間の割合も被験者間でほぼ一定だからである。

    Parameters
    ----------
    kind:
        "none"      そのまま（従来）
        "rank"      順位を [0,1] に写す（採用）
        "mad"       (x - 中央値) / (1.4826 * MAD)。録画全体。入力側と同じ式
        "mad_local" 上を 10 秒の移動窓で（時間変動する信号品質に追従する）
        "p99"       99 パーセンタイルで割る

    Notes
    -----
    **ピーク抽出だけでなく重心補正もこの正規化後の heatmap で行うこと。**
    重心は heatmap の値を重みに使うので、片方だけ正規化すると整合しない。
    順位変換は単調変換なので極大点の位置は変わらないが、重心の重みは変わる。
    """
    x = np.asarray(heatmap, dtype=np.float64)
    if kind == "none":
        return x
    if kind == "rank":
        order = np.argsort(np.argsort(x))
        return order / max(1, len(x) - 1)
    if kind == "mad":
        med = np.median(x)
        scale = 1.4826 * np.median(np.abs(x - med))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = x.std() if x.std() > 1e-9 else 1.0
        return (x - med) / scale
    if kind == "mad_local":
        w = int(10 * fs)
        step = max(1, w // 4)
        idx = np.arange(0, len(x), step)
        med = np.empty(len(idx))
        sc = np.empty(len(idx))
        for i, c in enumerate(idx):
            seg = x[max(0, c - w // 2):c + w // 2]
            m = np.median(seg)
            med[i] = m
            sc[i] = 1.4826 * np.median(np.abs(seg - m))
        sc[~np.isfinite(sc) | (sc < 1e-9)] = 1e-9
        g = np.arange(len(x))
        return (x - np.interp(g, idx, med)) / np.interp(g, idx, sc)
    if kind == "p99":
        p = np.percentile(x, 99)
        return x / (p if p > 1e-9 else 1.0)
    raise ValueError(f"unknown heatmap normalization: {kind}")
