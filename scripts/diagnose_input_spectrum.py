"""入力 RCG 50 点に呼吸成分がどれだけ残っているかを診断する (2026-08-25)。

動機 (CLAUDE.md R5「モデルより先に，入力と評価を疑う」):
後段の処理は一通り試し切り、線形しきい値が最良という結論が出た
(適応しきい値 / Viterbi 系列復号 / Kubios 拍系列補正 / ビート追跡 / 50ch 多数決 は
いずれも 11-fold LOSO 全体では改善しなかった)。残るのは前処理である。

Wang ら (Sensors 25(17):5607, 2025) は、ラグランジュ乗数を使った二次スペクトル疎分離で
呼吸成分と心拍成分を明示的に分離してから拍を取っている。呼吸は心拍より遥かに大振幅で、
しかもその高調波が心拍の周波数帯に重なることが知られている
(An innovative approach for FMCW radar vital sign monitoring with removal of
respiratory harmonics, Digital Signal Processing, 2024)。

我々の入力 (RCG 50 点) は Chen ら の 4D beamforming + K-means + micro-motion amplification を
経ているが、**呼吸成分が明示的に除かれているかどうかは論文本文からは読み取れない**。
まず実際に測る。対策を設計するのはその後。

測るもの (トライアルごと、50 チャネルの平均):
  - 呼吸帯 [0.1, 0.5] Hz のパワー比
  - 心拍基本波帯 [0.8, 3.0] Hz のパワー比
  - R 波の急峻さに効く高域 [3.0, 25.0] Hz のパワー比
  - 呼吸帯 / 心拍帯 のパワー比 (大きいほど呼吸に埋もれている)
  - 正解 ECG から測った実際の心拍数と、RCG のスペクトルピークが一致するか
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

RESP_BAND = (0.1, 0.5)
CARDIAC_BAND = (0.8, 3.0)
HIGH_BAND = (3.0, 25.0)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(psd[m].sum()) if m.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=str(REPO_ROOT / "data" / "raw"))
    ap.add_argument("--subjects", type=int, nargs="+",
                    default=[1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30])
    ap.add_argument("--max_trials_per_subject", type=int, default=3)
    args = ap.parse_args()

    print(f'{"被験者":>6s} {"試行":>5s} {"呼吸帯%":>8s} {"心拍帯%":>8s} {"高域%":>8s} '
          f'{"呼吸/心拍":>10s} {"RCGピーク":>9s} {"ECG HR":>8s} {"差":>7s}')
    rows = []
    for subj in args.subjects:
        tids = trial_ids_for_subjects(args.raw_root, [subj])[: args.max_trials_per_subject]
        for tid in tids:
            rec = load_trial(args.raw_root, tid)
            rcg = np.asarray(rec.rcg, dtype=float)      # (T, 50)
            fs = rec.fs

            # チャネルごとに PSD を取り、平均する (チャネル間で振幅が違うので正規化してから)
            psds = []
            for c in range(rcg.shape[1]):
                x = rcg[:, c]
                x = x - x.mean()
                sd = x.std()
                if sd < 1e-12:
                    continue
                x = x / sd
                spec = np.abs(np.fft.rfft(x)) ** 2
                psds.append(spec)
            if not psds:
                continue
            psd = np.mean(psds, axis=0)
            freqs = np.fft.rfftfreq(rcg.shape[0], d=1.0 / fs)

            total = band_power(freqs, psd, 0.05, 25.0)
            if total <= 0:
                continue
            p_resp = band_power(freqs, psd, *RESP_BAND) / total * 100.0
            p_card = band_power(freqs, psd, *CARDIAC_BAND) / total * 100.0
            p_high = band_power(freqs, psd, *HIGH_BAND) / total * 100.0
            ratio = p_resp / p_card if p_card > 0 else float("inf")

            # 心拍帯のスペクトルピーク (RCG から読める心拍数) と、正解 ECG の心拍数を比べる
            m = (freqs >= CARDIAC_BAND[0]) & (freqs < CARDIAC_BAND[1])
            rcg_hr = float(freqs[m][int(np.argmax(psd[m]))] * 60.0) if m.any() else float("nan")
            ecg = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
            peaks = detect_r_peaks_neurokit(ecg, fs)
            ecg_hr = float(60.0 * fs / np.median(np.diff(peaks))) if len(peaks) > 2 else float("nan")

            print(f"{subj:6d} {tid:5d} {p_resp:8.2f} {p_card:8.2f} {p_high:8.2f} "
                  f"{ratio:10.2f} {rcg_hr:9.1f} {ecg_hr:8.1f} {rcg_hr - ecg_hr:+7.1f}",
                  flush=True)
            rows.append((p_resp, p_card, p_high, ratio, abs(rcg_hr - ecg_hr)))

    if rows:
        a = np.asarray(rows, dtype=float)
        print("\n=== 平均 ===")
        print(f"  呼吸帯 [0.1,0.5) Hz : {a[:, 0].mean():.2f} %")
        print(f"  心拍帯 [0.8,3.0) Hz : {a[:, 1].mean():.2f} %")
        print(f"  高域   [3.0,25 ) Hz : {a[:, 2].mean():.2f} %")
        print(f"  呼吸/心拍 パワー比   : {a[:, 3].mean():.2f}")
        print(f"  RCG ピーク心拍数と ECG 心拍数の差の絶対値: {a[:, 4].mean():.1f} bpm")
        print("\n読み方: 呼吸/心拍 が 1 を大きく超えるなら、入力は呼吸に支配されており、")
        print("        呼吸成分の分離 (Wang ら Sensors 2025 の疎分離、VMD、SSA 等) に価値がある。")
        print("        1 を下回るなら、Chen ら の前処理で既に呼吸は落ちており、そこは伸びしろではない。")


if __name__ == "__main__":
    main()
