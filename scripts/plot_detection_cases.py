"""検出がうまくいっている例と失敗している例を、生波形から正解まで縦に並べて描く (2026-08-25)。

目的: 数値だけでは「なぜ落ちているか」が分からないので、目視で診断する。
1 枚につき 1 トライアルの一区間を、上から順に

  1. レーダ RCG 50 点 (時間 x 点のヒートマップ)
  2. 分散が最大のチャネルの生波形
  3. モデルが出した heatmap と、しきい値で拾ったピーク (赤), 正解 R 位置 (緑破線)
  4. 正解 ECG と、その R 位置 (緑)

で描く。2 と 3 と 4 の時間軸は揃えてある。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path("/home/shunya/research/radarODE-MTL/scripts")))

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.rpeaks import (  # noqa: E402
    detect_r_peaks_neurokit,
    extract_peaks_from_heatmap,
    match_peaks,
    refine_peaks_centroid,
)
from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN  # noqa: E402

# 日本語フォント: fc-list で見えている ttc を matplotlib に直接登録する
# (findfont はフォールバックして DejaVu Sans を返してしまい、日本語が豆腐になる)
for _p in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"):
    if Path(_p).exists():
        try:
            matplotlib.font_manager.fontManager.addfont(_p)
            plt.rcParams["font.family"] = matplotlib.font_manager.FontProperties(
                fname=_p).get_name()
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False

WINDOW_SEC = 4.0


def zscore_channels(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / np.where(sd < 1e-8, 1.0, sd)


@torch.no_grad()
def predict_continuous(model, rcg_z, posxyz, win, step, device):
    """4 秒窓をずらして推論し、重なりを平均して連続の heatmap にする。"""
    n = rcg_z.shape[0]
    out = np.zeros(n, dtype=float)
    cnt = np.zeros(n, dtype=float)
    pos = torch.tensor(posxyz, dtype=torch.float32, device=device).unsqueeze(0)
    for s in range(0, max(1, n - win + 1), step):
        e = s + win
        if e > n:
            break
        seg = torch.tensor(rcg_z[s:e], dtype=torch.float32, device=device).unsqueeze(0)
        pred = model(seg, pos).squeeze(0).cpu().numpy()
        out[s:e] += pred
        cnt[s:e] += 1.0
    return out / np.where(cnt < 1, 1.0, cnt)


def plot_one(rec, pred, true_peaks, pred_peaks, t0_sec, dur_sec, title, out_path):
    fs = rec.fs
    a, b = int(t0_sec * fs), int((t0_sec + dur_sec) * fs)
    b = min(b, len(pred), len(rec.ecg))
    t = np.arange(a, b) / fs

    rcg = np.asarray(rec.rcg, dtype=float)[a:b]          # (T, 50)
    rcg_z = zscore_channels(rcg)
    var_ch = int(np.argmax(rcg.var(axis=0)))

    tp = true_peaks[(true_peaks >= a) & (true_peaks < b)]
    pp = np.asarray(pred_peaks)
    pp = pp[(pp >= a) & (pp < b)]

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=False,
                             gridspec_kw={"height_ratios": [1.4, 1.0, 1.2, 1.0]})

    ax = axes[0]
    v = np.percentile(np.abs(rcg_z), 98)
    ax.imshow(rcg_z.T, aspect="auto", origin="lower", cmap="RdBu_r",
              vmin=-v, vmax=v, extent=[t[0], t[-1], 0, rcg_z.shape[1]])
    for p in tp:
        ax.axvline(p / fs, color="limegreen", lw=0.7, alpha=0.55)
    ax.set_ylabel("レーダ 50 点\n(z 化)")
    ax.set_title(title, fontsize=12)

    ax = axes[1]
    ax.plot(t, rcg_z[:, var_ch], color="0.25", lw=0.7)
    for p in tp:
        ax.axvline(p / fs, color="limegreen", lw=0.7, alpha=0.55)
    ax.set_ylabel(f"分散最大ch\n(ch {var_ch})")
    ax.margins(x=0)

    ax = axes[2]
    ax.plot(t, pred[a:b], color="tab:blue", lw=1.0, label="モデル出力 (heatmap)")
    ax.axhline(0.3, color="orange", ls=":", lw=1.2, label="しきい値 0.3")
    for i, p in enumerate(tp):
        ax.axvline(p / fs, color="limegreen", lw=1.0, ls="--", alpha=0.8,
                   label="正解 R 位置" if i == 0 else None)
    for i, p in enumerate(pp):
        # 重心補正後のピーク位置は小数になるので、値を読むときだけ整数に丸める
        ax.plot(p / fs, pred[int(round(float(p)))], "v", color="red", ms=7,
                label="検出したピーク" if i == 0 else None)
    ax.set_ylim(-0.02, max(0.35, float(np.max(pred[a:b])) * 1.1))
    ax.set_ylabel("モデル出力")
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.margins(x=0)

    ax = axes[3]
    ecg = np.asarray(rec.ecg, dtype=float)[a:b]
    ax.plot(t, ecg, color="0.15", lw=0.8)
    for p in tp:
        ax.axvline(p / fs, color="limegreen", lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel("正解 ECG")
    ax.set_xlabel("時間 [s]")
    ax.margins(x=0)

    for a_ in axes[1:]:
        a_.set_xlim(t[0], t[-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"保存: {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO_ROOT / "reports" / "spatial_gnn_lodo_cv_runs.json"))
    ap.add_argument("--raw_root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--subjects", type=int, nargs="+", default=[17, 1, 16, 9])
    ap.add_argument("--height", type=float, default=0.3)
    ap.add_argument("--t0", type=float, default=20.0)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--outdir", default=str(REPO_ROOT / "reports" / "detection_cases"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    folds = {r["test_subject"]: r for r in json.load(open(args.runs)) if r.get("checkpoint_exists")}

    for subj in args.subjects:
        if subj not in folds:
            print(f"被験者 {subj}: チェックポイントなし")
            continue
        model = RPeakSpatialGNN(n_points=50, embed_dim=32, k_neighbors=6, n_gat_layers=2).to(device)
        model.load_state_dict(torch.load(folds[subj]["checkpoint"], map_location=device,
                                         weights_only=False))
        model.eval()

        tid = trial_ids_for_subjects(args.raw_root, [subj])[0]
        rec = load_trial(args.raw_root, tid)
        rcg_z = zscore_channels(np.asarray(rec.rcg, dtype=float))
        ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
        true_peaks = detect_r_peaks_neurokit(ecg, rec.fs)

        pred = predict_continuous(model, rcg_z, rec.posxyz,
                                  int(WINDOW_SEC * rec.fs), int(0.25 * rec.fs), device)
        true_peaks = true_peaks[true_peaks < len(pred)]
        pk = extract_peaks_from_heatmap(pred, rec.fs, height=args.height)
        if len(pk):
            pk = refine_peaks_centroid(pred, pk, rec.fs, half_win_ms=85.0, n_iter=2)
        m = match_peaks(true_peaks, pk, rec.fs, tolerance_ms=150.0)

        title = (f"被験者 {subj} / トライアル {tid}   "
                 f"F1(許容150ms)={m['f1']:.3f}  適合率={m['precision']:.3f}  再現率={m['recall']:.3f}   "
                 f"正解 {m['n_true']} 拍 / 検出 {m['n_pred']} 拍")
        plot_one(rec, pred, true_peaks, pk, args.t0, args.dur, title,
                 outdir / f"subject{subj:02d}_trial{tid}.png")


if __name__ == "__main__":
    main()
