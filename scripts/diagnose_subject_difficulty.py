"""11-fold LOSO で被験者ごとに F1 が 0.113〜0.906 と大きく振れた原因を、入力側から診断する。

なぜ入力側から見るか(2026-08-23):
    CLAUDE.md R5 のとおり、これまで価値が残った知見はすべて前処理・時間分解能・評価指標の側から
    出ており、モデル側の工夫(複素NN・NCP・容量拡大・二段構成)はほぼ全滅している。
    「subject 9 で F1 0.113 まで落ちる」という現象も、まずレーダ信号そのものの性質で
    説明できるかを確かめる。

出す量(すべて学習を使わず、生データだけから計算する):
    cardiac_max / cardiac_mean : 心拍帯[0.8,3]Hz のパワー比。50点それぞれについて計算し、
                                 最大値と平均値を取る。レーダが心拍由来の変位を
                                 どれだけ拾えているかの目安(SNR代理)
    hr_mean / rr_std           : 正解ECG(NeuroKit2)から求めた平均心拍数とRR間隔の標準偏差
    ecg_nan_frac               : 参照ECGの欠損割合
    n_trials                   : その被験者のトライアル数

最後に、各量と LOSO の F1 との Spearman 相関を出す。
相関が出れば「この条件では取れない」と条件つきで言えるようになり、
出なければ入力の素性では説明できない(モデル側または別の要因)ことが分かる。

使い方:
    .venv/bin/python scripts/diagnose_subject_difficulty.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.channel_weighting import cardiac_band_power_ratio  # noqa: E402
from dog_radar_vitals.data.rpeaks import detect_r_peaks_neurokit  # noqa: E402

ALL_SUBJECTS = [1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30]

# reports/spatial_gnn_lodo_cv_eval.log から転記した 11-fold LOSO の実測値
LOSO_F1 = {1: 0.414, 2: 0.653, 5: 0.244, 9: 0.113, 10: 0.743, 13: 0.793,
           14: 0.548, 16: 0.443, 17: 0.906, 29: 0.590, 30: 0.787}
LOSO_RR = {1: 9.99, 2: 8.25, 5: 8.60, 9: 14.29, 10: 13.36, 13: 11.89,
           14: 11.26, 16: 8.58, 17: 9.31, 29: 8.72, 30: 11.22}

OUT_PATH = REPO_ROOT / "reports/subject_difficulty_diagnosis.json"


def spearman(x, y):
    """順位相関(scipy を使わず順位のピアソン相関で計算する)。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    raw_root = REPO_ROOT / "data/raw"
    rows = []
    for subj in ALL_SUBJECTS:
        trial_ids = trial_ids_for_subjects(raw_root, [subj])
        cmax, cmean, hrs, rrstds, nanfracs = [], [], [], [], []
        for tid in trial_ids:
            rec = load_trial(raw_root, tid)
            scores = np.array([cardiac_band_power_ratio(rec.rcg[:, ch], rec.fs)
                               for ch in range(rec.rcg.shape[1])])
            cmax.append(float(scores.max()))
            cmean.append(float(scores.mean()))
            nanfracs.append(float(np.isnan(rec.ecg).mean()))
            ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
            peaks = detect_r_peaks_neurokit(ecg, rec.fs)
            if len(peaks) >= 3:
                rr = np.diff(peaks) / rec.fs * 1000.0
                hrs.append(60000.0 / float(np.mean(rr)))
                rrstds.append(float(np.std(rr)))
        rows.append({
            "subject": subj, "n_trials": len(trial_ids),
            "cardiac_max": float(np.mean(cmax)), "cardiac_mean": float(np.mean(cmean)),
            "hr_mean": float(np.mean(hrs)) if hrs else float("nan"),
            "rr_std": float(np.mean(rrstds)) if rrstds else float("nan"),
            "ecg_nan_frac": float(np.mean(nanfracs)),
            "loso_f1": LOSO_F1[subj], "loso_rr_mae": LOSO_RR[subj],
        })
        r = rows[-1]
        print(f"subject {subj:>3}: n={r['n_trials']:>2}  cardiac_max={r['cardiac_max']:.4f}  "
              f"cardiac_mean={r['cardiac_mean']:.4f}  HR={r['hr_mean']:.1f}  "
              f"rr_std={r['rr_std']:.1f}ms  ecg_nan={r['ecg_nan_frac']:.3f}  "
              f"F1={r['loso_f1']:.3f}", flush=True)

    print("\n=== LOSO の F1 との Spearman 順位相関 (n=11) ===")
    corrs = {}
    for key in ("cardiac_max", "cardiac_mean", "hr_mean", "rr_std", "ecg_nan_frac", "n_trials"):
        rho = spearman([r[key] for r in rows], [r["loso_f1"] for r in rows])
        corrs[key] = rho
        print(f"  {key:14s} vs F1 : rho = {rho:+.3f}")
    print("\n=== LOSO の RR間隔MAE との Spearman 順位相関 ===")
    for key in ("cardiac_max", "cardiac_mean", "hr_mean", "rr_std", "n_trials"):
        rho = spearman([r[key] for r in rows], [r["loso_rr_mae"] for r in rows])
        corrs[key + "__rr"] = rho
        print(f"  {key:14s} vs RR-MAE : rho = {rho:+.3f}")
    print("\n注意: n=11 なので |rho| が 0.6 程度でも偶然の範囲に入りうる。"
          "強い相関が出た量だけを、条件つきの説明として扱う。")

    OUT_PATH.write_text(json.dumps({"rows": rows, "spearman": corrs}, indent=2, ensure_ascii=False))
    print(f"\n保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
