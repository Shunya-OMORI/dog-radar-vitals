"""レーダを一切見ずに、正解 R 波の「過去」だけから次の R 波位置を当てるとどこまで行けるか (2026-08-25)。

動機:
Chen ら (IEEE T-MC 23(1):270-285, 2024) の式(13) は
    p(X|h) = prod_t p(x_t | x_1, ..., x_{t-1}, h_t)
であり、本文 3.3 に「dilated convolutional blocks that **receives X_{1:T} and h_t as inputs**」と
明記されている。つまりネットワークは **ECG そのものの過去**を入力に取る自己回帰モデルである。
本文 4.2.2 は「学習時は並列 (step = 640 - 512 = 128)、推論時は autoregressive で 1 サンプルずつ」
と書くが、**推論を始めるときの最初の 512 サンプル (2.56 秒) の文脈をどう与えるかは書かれていない**。
公開リポジトリ (jinbochen0823/RCG2ECG) にはデータセットのみでネットワークのコードがなく、確認できない。

もし初期文脈に正解 ECG を与えているなら、直前 2〜3 拍の R 波位置が既知ということになり、
心拍が準周期的である以上、次の R 波位置はかなりの精度で決まってしまう。
彼らが報告する R 波タイミング中央値 3 ms は、その範囲で説明がつくかもしれない。

そこで **レーダを一切使わず、正解 R 波の過去だけを見る予測器**の誤差を測る。
これが 3 ms 前後なら「周期性だけで到達できる水準」ということになり、
レーダから情報を取れている証拠にはならない。

予測方式 (いずれも「次の 1 拍」を、それより前の正解 R 波位置だけから予測):
  last     : 直前の RR 間隔をそのまま次の RR 間隔とする
  meanN    : 直前 N 拍の RR 間隔の平均
  medianN  : 直前 N 拍の RR 間隔の中央値

これは Chen ら の手法の再現ではない。**「過去の正解を見てよい」という条件下で、
周期性だけからどこまで当たるかの上限の目安**である。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.rpeaks import detect_r_peaks_neurokit  # noqa: E402


def predict_errors(peaks: np.ndarray, fs: int, mode: str, n: int) -> np.ndarray:
    """peaks[i] を peaks[:i] だけから予測したときの誤差 [ms] を返す。"""
    t = np.asarray(peaks, dtype=float) / fs * 1000.0
    rr = np.diff(t)
    errs = []
    for i in range(max(2, n), len(t)):
        hist = rr[max(0, i - 1 - n): i - 1]      # 直前 n 個の RR 間隔 (i-1 番目までの情報のみ)
        if hist.size == 0:
            continue
        if mode == "last":
            step = rr[i - 2]
        elif mode == "mean":
            step = float(np.mean(hist))
        else:
            step = float(np.median(hist))
        errs.append((t[i - 1] + step) - t[i])
    return np.asarray(errs, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--subjects", type=int, nargs="+",
                    default=[1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30])
    ap.add_argument("--max_trials_per_subject", type=int, default=3)
    args = ap.parse_args()

    modes = [("last", 1), ("mean", 3), ("median", 3), ("mean", 8), ("median", 8)]
    acc: dict[str, list[np.ndarray]] = {f"{m}{n}": [] for m, n in modes}

    for subj in args.subjects:
        for tid in trial_ids_for_subjects(args.raw_root, [subj])[: args.max_trials_per_subject]:
            rec = load_trial(args.raw_root, tid)
            ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
            peaks = detect_r_peaks_neurokit(ecg, rec.fs)
            if len(peaks) < 12:
                continue
            for m, n in modes:
                e = predict_errors(peaks, rec.fs, m, n)
                if e.size:
                    acc[f"{m}{n}"].append(e)

    print("レーダを一切見ず、正解 R 波の過去だけから次の R 波位置を予測した場合の誤差")
    print(f"{'予測方式':>12s} {'平均|誤差|':>10s} {'中央値|誤差|':>12s} {'90%tile':>9s} {'標準偏差':>9s}")
    for key, chunks in acc.items():
        if not chunks:
            continue
        e = np.concatenate(chunks)
        print(f"{key:>12s} {np.mean(np.abs(e)):10.2f} {np.median(np.abs(e)):12.2f} "
              f"{np.percentile(np.abs(e), 90):9.2f} {np.std(e):9.2f}")
    print()
    print("参考: Chen ら が報告する R 波タイミング誤差は 中央値 3 ms / 90 パーセンタイル 9 ms")
    print("     （Q 14 ms / S 8 ms / T 10 ms なので、R だけが突出して良い）")
    print("この表の中央値が 3 ms を大きく上回るなら、周期性だけでは 3 ms に届かないことになり、")
    print("彼らはレーダから実際に情報を取れていると考えてよい。")
    print("逆に 3 ms 前後まで下がるなら、過去の正解 ECG を推論時に与えているかどうかで")
    print("値の意味が変わるため、同じ表に並べる前に確認が要る。")


if __name__ == "__main__":
    main()
