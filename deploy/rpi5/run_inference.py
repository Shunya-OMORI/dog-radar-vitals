"""Raspberry Pi 5 向けの最小推論エントリポイント (2026-08-07)。

確定構成(2026-08-07): 空間GNN config284 epoch4 (sigma=15ms教師, NeuroKit2ラベル)
+ 固定しきい値0.3のピーク検出 + サブサンプル重心復号 refine_peaks_centroid(±85ms, 2回)
= F1 0.749 / RR-MAE 8.41ms / RMSSD-MAE 10.27ms (WSL側 test 3被験者44トライアルで確認済み)

処理: 1トライアル分のRCG(.mat)読み込み -> 窓内z-score -> 4秒窓・0.25秒ステップの
スライディング窓推論(重複平均で連続heatmap化) -> しきい値0.3 + 重心復号 -> R波時刻列
-> RR間隔・RMSSD。レイテンシ(前処理・1窓あたり推論・後処理)を計測して表示する。

依存: torch(CPU), torch_geometric(pure-Python wheel; torch-scatter等の
コンパイル拡張は不要なことをWSL側で確認済み), numpy, scipy のみ。
--gt を付けた場合のみ参照ECGとの比較にneurokit2を使う(精度検証用、Piでは任意)。

使い方(リポジトリ直下から):
  python deploy/rpi5/run_inference.py --trial 48                # 精度検証つき
  python deploy/rpi5/run_inference.py --trial 48 --no-gt        # 推論+レイテンシのみ
  python deploy/rpi5/run_inference.py --trial 48 --threads 4
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dog_radar_vitals.data.mmecg import load_trial  # noqa: E402
from dog_radar_vitals.data.mmecg_windowing import zscore_channels  # noqa: E402
from dog_radar_vitals.data.rpeaks import (  # noqa: E402
    extract_peaks_from_heatmap,
    match_peaks,
    matched_rr_interval_mae_ms,
    refine_peaks_centroid,
)
from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN  # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), 'checkpoints', '284_sigma15_epoch004.pt')
WINDOW_SEC, STEP_SEC = 4.0, 0.25
PEAK_HEIGHT, CENTROID_MS, CENTROID_ITERS = 0.3, 85.0, 2


def clean_rr_intervals_ms(peaks, fs, min_rr_ms=300.0, max_rr_ms=1500.0, rel_dev=0.2):
    """標準的なHRVアーチファクト除去(radarODE-MTL/scripts/evaluate_anchor_hrv_metrics.py
    と同一実装のコピー; Pi側にradarODE-MTLを置かないための重複)。"""
    if len(peaks) < 3:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000.0
    keep = (rr >= min_rr_ms) & (rr <= max_rr_ms)
    rr_valid = rr[np.where(keep)[0]]
    if len(rr_valid) < 3:
        return rr_valid
    med = np.median(rr_valid)
    return rr_valid[np.abs(rr_valid - med) <= rel_dev * med]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', type=int, required=True, help='MMECG trial id (テスト被験者は48〜91)')
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--raw_root', default=os.path.join(REPO_ROOT, 'data', 'raw'))
    ap.add_argument('--threads', type=int, default=4, help='CPUスレッド数 (RPi5は4コア)')
    ap.add_argument('--no-gt', action='store_true', help='参照ECGとの比較を省略 (neurokit2不要になる)')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device('cpu')

    model = RPeakSpatialGNN(n_points=50, embed_dim=32, k_neighbors=6, n_gat_layers=2, n_downsample=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: RPeakSpatialGNN ({n_params/1e6:.3f} M params), checkpoint={os.path.basename(args.checkpoint)}')
    print(f'threads={args.threads}, device=cpu')

    rec = load_trial(args.raw_root, args.trial)
    fs = rec.fs
    window_len, step_len = int(WINDOW_SEC * fs), int(STEP_SEC * fs)

    t0 = time.perf_counter()
    rcg_z = zscore_channels(rec.rcg)
    t_pre = time.perf_counter() - t0

    n_steps = rcg_z.shape[0]
    fused_sum = np.zeros(n_steps, dtype=np.float64)
    fused_count = np.zeros(n_steps, dtype=np.float64)
    posxyz_t = torch.from_numpy(rec.posxyz).float().unsqueeze(0)
    starts = list(range(0, n_steps - window_len + 1, step_len))
    lat = []
    with torch.no_grad():
        for s in starts:
            tw = time.perf_counter()
            x = torch.from_numpy(rcg_z[s:s + window_len]).float().unsqueeze(0)
            pred = model(x, posxyz_t).numpy().squeeze(0)
            lat.append(time.perf_counter() - tw)
            fused_sum[s:s + window_len] += pred
            fused_count[s:s + window_len] += 1.0
    fused_count[fused_count == 0] = 1.0
    heatmap = (fused_sum / fused_count).astype(np.float32)

    t1 = time.perf_counter()
    peaks_int = extract_peaks_from_heatmap(heatmap, fs, height=PEAK_HEIGHT)
    peaks = refine_peaks_centroid(heatmap, peaks_int, fs, half_win_ms=CENTROID_MS, n_iter=CENTROID_ITERS)
    t_post = time.perf_counter() - t1

    rr_clean = clean_rr_intervals_ms(peaks, fs)
    rmssd = float(np.sqrt(np.mean(np.diff(rr_clean) ** 2))) if len(rr_clean) >= 3 else float('nan')

    lat = np.array(lat) * 1000.0
    dur_sec = n_steps / fs
    print(f'\ntrial {args.trial}: {dur_sec:.0f}s, {len(starts)} windows (step {STEP_SEC}s)')
    print(f'detected R peaks: {len(peaks)}  mean HR: {60.0 / (np.mean(rr_clean) / 1000.0):.1f} bpm'
          if len(rr_clean) else f'detected R peaks: {len(peaks)}')
    print(f'RMSSD (radar): {rmssd:.2f} ms')
    print('\n--- latency (CPU) ---')
    print(f'preprocess (z-score whole trial): {t_pre*1000:.1f} ms')
    print(f'inference per 4s-window: mean {lat.mean():.1f} ms, p50 {np.percentile(lat, 50):.1f} ms, '
          f'p95 {np.percentile(lat, 95):.1f} ms  (window budget {WINDOW_SEC*1000:.0f} ms, '
          f'occupancy {lat.mean()/ (STEP_SEC*1000) * 100:.1f}% at step {STEP_SEC}s)')
    print(f'postprocess (peaks+centroid, whole trial): {t_post*1000:.1f} ms')

    if not args.no_gt:
        from dog_radar_vitals.data.rpeaks import detect_r_peaks_neurokit
        ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
        true_peaks = detect_r_peaks_neurokit(ecg, fs)
        true_peaks = true_peaks[true_peaks < len(heatmap)]
        m = match_peaks(true_peaks, peaks, fs, tolerance_ms=50)
        rr_mae = matched_rr_interval_mae_ms(m, fs)
        true_rr = clean_rr_intervals_ms(true_peaks, fs)
        true_rmssd = float(np.sqrt(np.mean(np.diff(true_rr) ** 2))) if len(true_rr) >= 3 else float('nan')
        print('\n--- accuracy vs reference ECG (NeuroKit2) ---')
        print(f'F1={m["f1"]:.3f} precision={m["precision"]:.3f} recall={m["recall"]:.3f} '
              f'RR-MAE={rr_mae:.2f} ms  RMSSD err={abs(rmssd - true_rmssd):.2f} ms '
              f'(radar {rmssd:.2f} vs ECG {true_rmssd:.2f})')


if __name__ == '__main__':
    main()
