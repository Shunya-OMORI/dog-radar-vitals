"""発表資料用の研究標準図を、現時点の最高性能構成で生成する (2026-08-07)。

構成: 空間GNN config284 epoch4 (sigma=15ms教師) + 固定しきい値0.3のピーク検出
      + サブサンプル重心復号 refine_peaks_centroid(±85ms, 2回)
      = F1 0.749 / RR-MAE 8.41ms / RMSSD-MAE 10.27ms (test 3被験者44トライアル)

生成する図 (reports/mmecg_comparison/):
  1. pr_curve_284ep4.png            検出しきい値掃引によるPrecision-Recall曲線
  2. rpeak_detection_example_284ep4.png  参照ECG+正解R波 vs 予測heatmap+検出R波の定性例
  3. rr_blandaltman_284ep4.png      マッチ済みRR間隔の散布図 + Bland-Altman
入力heatmapは compare_peak_postprocessing.py が保存したキャッシュ
(radarODE-MTL/reports/peak_postprocessing_heatmap_cache_284ep4.npz)を使う。
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DOG_RADAR_ROOT = '/home/shunya/research/dog-radar-vitals'
sys.path.insert(0, os.path.join(DOG_RADAR_ROOT, 'src'))

from dog_radar_vitals.data.mmecg import load_trial  # noqa: E402
from dog_radar_vitals.data.rpeaks import (  # noqa: E402
    extract_peaks_from_heatmap,
    match_peaks,
    refine_peaks_centroid,
)

CACHE = '/home/shunya/research/radarODE-MTL/reports/peak_postprocessing_heatmap_cache_284ep4.npz'
OUT_DIR = os.path.join(DOG_RADAR_ROOT, 'reports', 'mmecg_comparison')

# dataviz: validated categorical palette (light surface)
C_BLUE = '#2a78d6'    # 予測 (モデル出力)
C_ORANGE = '#eb6834'  # 検出R波 (後処理出力)
C_INK = '#0b0b0b'
C_MUT = '#52514e'
GRID = dict(color='#e5e4e0', linewidth=0.8)

plt.rcParams.update({
    'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
    'axes.edgecolor': '#c9c8c2', 'axes.linewidth': 0.8,
    'axes.labelcolor': C_INK, 'text.color': C_INK,
    'xtick.color': C_MUT, 'ytick.color': C_MUT,
    'font.size': 10, 'axes.titlesize': 11,
})


def detect(pred, fs, height=0.3):
    peaks = extract_peaks_from_heatmap(pred, fs, height=height)
    return refine_peaks_centroid(pred, peaks, fs, half_win_ms=85.0, n_iter=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(CACHE, allow_pickle=False)
    cache = {k: data[k] for k in data.files}
    trial_ids = [int(t) for t in cache['trial_ids']]

    # ---------- 全トライアルの検出・マッチ (運用しきい値0.3) ----------
    per_trial = {}
    for tid in trial_ids:
        pred, true_pk = cache[f'{tid}_pred'], cache[f'{tid}_true']
        fs = int(cache[f'{tid}_fs'][0])
        pp = detect(pred, fs)
        tt = true_pk[true_pk < len(pred)]
        per_trial[tid] = (pred, tt, pp, fs, match_peaks(tt, pp, fs, tolerance_ms=50))

    # ---------- 図1: Precision-Recall曲線 (しきい値掃引) ----------
    heights = np.round(np.arange(0.05, 0.91, 0.025), 3)
    precs, recs = [], []
    for h in heights:
        ps, rs = [], []
        for tid in trial_ids:
            pred, tt = cache[f'{tid}_pred'], None
            fs = int(cache[f'{tid}_fs'][0])
            true_pk = cache[f'{tid}_true']
            tt = true_pk[true_pk < len(pred)]
            pp = detect(pred, fs, height=h)
            m = match_peaks(tt, pp, fs, tolerance_ms=50)
            ps.append(m['precision']); rs.append(m['recall'])
        precs.append(np.mean(ps)); recs.append(np.mean(rs))
    precs, recs = np.array(precs), np.array(recs)

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    # faint iso-F1 contours (standard in detection papers)
    gr = np.linspace(0.01, 1, 200)
    for f1v in [0.3, 0.5, 0.7, 0.9]:
        pv = f1v * gr / np.maximum(2 * gr - f1v, 1e-9)
        mask = (pv > 0) & (pv <= 1) & (gr >= f1v / 2)
        ax.plot(gr[mask], pv[mask], color='#e5e4e0', lw=0.8, zorder=1)
        ax.annotate(f'F1={f1v}', xy=(0.985, f1v * 0.985 / np.maximum(2 * 0.985 - f1v, 1e-9)),
                    fontsize=7, color='#9a9992', ha='right', va='bottom')
    ax.plot(recs, precs, color=C_BLUE, lw=2, zorder=3)
    i_op = int(np.argmin(np.abs(heights - 0.3)))
    ax.scatter([recs[i_op]], [precs[i_op]], s=55, color=C_ORANGE, zorder=4)
    ax.annotate(f'operating point\n(threshold=0.3)\nP={precs[i_op]:.2f}, R={recs[i_op]:.2f}',
                xy=(recs[i_op], precs[i_op]), xytext=(recs[i_op] - 0.38, precs[i_op] - 0.18),
                fontsize=8, color=C_INK,
                arrowprops=dict(arrowstyle='-', color=C_MUT, lw=0.8))
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.0)
    ax.set_title('R-peak detection: precision–recall\n(spatial GNN + centroid decoding, 44 test trials)')
    ax.grid(**GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'pr_curve_284ep4.png'))
    plt.close(fig)
    print('saved pr_curve_284ep4.png')

    # ---------- 図2: 定性例 (F1中央値のトライアル、10秒) ----------
    f1s = {tid: per_trial[tid][4]['f1'] for tid in trial_ids}
    med_tid = sorted(trial_ids, key=lambda t: f1s[t])[len(trial_ids) // 2]
    pred, tt, pp, fs, m = per_trial[med_tid]
    rec = load_trial(os.path.join(DOG_RADAR_ROOT, 'data', 'raw'), med_tid)
    ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))

    t0, dur = 30.0, 10.0  # 30〜40秒の10秒区間
    s0, s1 = int(t0 * fs), int((t0 + dur) * fs)
    tax = np.arange(s0, s1) / fs
    ecg_seg = ecg[s0:s1]
    ecg_seg = (ecg_seg - ecg_seg.mean()) / (ecg_seg.std() + 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.0), dpi=200, sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    ax = axes[0]
    ax.plot(tax, ecg_seg, color=C_MUT, lw=1.0)
    tt_seg = tt[(tt >= s0) & (tt < s1)]
    ax.scatter(tt_seg / fs, ecg_seg[tt_seg - s0] + 0.4, marker='v', s=30, color=C_INK, zorder=3,
               label='reference R-peak (NeuroKit2)')
    ax.set_ylabel('reference ECG\n(z-scored)')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(**GRID); ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(tax, pred[s0:s1], color=C_BLUE, lw=1.4, label='predicted heatmap')
    pp_seg = pp[(pp >= s0) & (pp < s1)]
    for i, p in enumerate(pp_seg):
        ax.axvline(p / fs, color=C_ORANGE, lw=1.2, alpha=0.9, zorder=2,
                   label='detected R-peak (threshold 0.3\n+ centroid refinement)' if i == 0 else None)
    ax.axhline(0.3, color=C_MUT, lw=0.8, ls=(0, (4, 3)))
    ax.annotate('threshold 0.3', xy=(tax[-1], 0.3), fontsize=7, color=C_MUT,
                ha='right', va='bottom')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('R-peak heatmap')
    ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    ax.grid(**GRID); ax.set_axisbelow(True)
    axes[0].set_title(f'Radar-only R-peak detection vs reference ECG '
                      f'(test trial {med_tid}, median-F1 example, F1={f1s[med_tid]:.2f})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'rpeak_detection_example_284ep4.png'))
    plt.close(fig)
    print(f'saved rpeak_detection_example_284ep4.png (trial {med_tid}, F1={f1s[med_tid]:.3f})')

    # ---------- 図3: RR散布図 + Bland-Altman ----------
    true_rr_all, pred_rr_all = [], []
    for tid in trial_ids:
        _, _, _, fs, m = per_trial[tid]
        mt, mp = m['matched_true'], m['matched_pred']
        if len(mt) < 2:
            continue
        true_rr_all.append(np.diff(mt) / fs * 1000.0)
        pred_rr_all.append(np.diff(mp) / fs * 1000.0)
    true_rr = np.concatenate(true_rr_all)
    pred_rr = np.concatenate(pred_rr_all)
    keep = (true_rr >= 300) & (true_rr <= 1500)  # 隣接マッチが飛んだ区間は除外
    true_rr, pred_rr = true_rr[keep], pred_rr[keep]
    diff = pred_rr - true_rr
    mean_ = (pred_rr + true_rr) / 2
    bias, sd = float(diff.mean()), float(diff.std())
    mae = float(np.abs(diff).mean())
    r = float(np.corrcoef(true_rr, pred_rr)[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), dpi=200)
    ax = axes[0]
    lim = (300, 1200)
    ax.plot(lim, lim, color='#c9c8c2', lw=1.0, zorder=1)
    ax.scatter(true_rr, pred_rr, s=5, alpha=0.25, color=C_BLUE, edgecolors='none', zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('reference RR interval [ms]')
    ax.set_ylabel('radar-estimated RR interval [ms]')
    ax.set_title(f'RR agreement (n={len(true_rr)} intervals)\nr={r:.3f}, MAE={mae:.1f} ms')
    ax.grid(**GRID); ax.set_axisbelow(True)

    ax = axes[1]
    ax.scatter(mean_, diff, s=5, alpha=0.25, color=C_BLUE, edgecolors='none', zorder=2)
    ax.axhline(bias, color=C_INK, lw=1.2)
    for y, lab in [(bias + 1.96 * sd, '+1.96 SD'), (bias - 1.96 * sd, '-1.96 SD')]:
        ax.axhline(y, color=C_ORANGE, lw=1.0, ls=(0, (4, 3)))
        ax.annotate(f'{lab} = {y:+.1f} ms', xy=(1160, y), fontsize=8, color=C_INK,
                    ha='right', va='bottom')
    ax.annotate(f'bias = {bias:+.1f} ms', xy=(1160, bias), fontsize=8, color=C_INK,
                ha='right', va='bottom')
    ax.set_xlim(300, 1200)
    ax.set_xlabel('mean of reference & estimate [ms]')
    ax.set_ylabel('estimate - reference [ms]')
    ax.set_title('Bland–Altman (RR intervals)')
    ax.grid(**GRID); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'rr_blandaltman_284ep4.png'))
    plt.close(fig)
    print(f'saved rr_blandaltman_284ep4.png (bias={bias:+.2f}, sd={sd:.2f}, mae={mae:.2f}, r={r:.3f})')


if __name__ == '__main__':
    main()
