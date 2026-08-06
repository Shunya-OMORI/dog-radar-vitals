"""CASEデータセットで「HR由来の覚醒度スコア」が連続arousalアノテーションと相関するかを検証する。

## この評価の位置付け(research/CLAUDE.md R3)

新しい指標(覚醒度スコア)を作ったら、モデル学習より先に「既知の値を再現できるか」を確認する
のがこのプロジェクトのルールである。ここでは学習済みモデルの代わりに、
`dog_radar_vitals.affect.arousal_from_hr`の信号処理パイプラインそのものを検証対象とする。

**参考値(厳密な再現対象ではない)**: このプロジェクトの過去の記憶(MEMORY.md)によれば、
WESADデータセットで「覚醒度R^2≈0.55」という結果が過去に得られている。CASEとWESADは
被験者・刺激・アノテーション方式が異なるデータセットであり、直接比較はできないが、
「HR由来の覚醒度スコアが連続アノテーションとどの程度の相関を持ちうるか」のオーダー感の
参考として記録しておく。相関が大きく異なっても(低くても高くても)、それ自体が結果であり、
無理にこの値に近づける調整はしない。

## 評価プロトコル

1. 被験者ごとにECGからR波を検出し(`arousal_from_hr.rpeaks_from_ecg`)、RR間隔系列を得る。
2. 非重複10秒窓(research/CLAUDE.md制約4: 重複窓禁止)でHRを集計し、
   その被験者の**セッション冒頭の`startVid`区間**(全30名で共通に最初に提示される、
   Great Barrier Reefの穏やかな導入映像。詳細は`data/case/NOTES.md`参照)をベースラインとして
   z-score正規化する。CASEには明示的な"baseline"ラベルは無いため、`metadata/seqs_order.txt`
   で全被験者が最初に見ることを確認した`startVid`区間を安静時ベースラインとして採用する
   (CASEの実験デザイン上、感情喚起刺激の提示前であるため)。
3. 同じ非重複10秒窓でCASEの連続arousalアノテーション(interpolated, 元は20Hzジョイスティック)
   を窓内平均し、HR由来の覚醒度スコアとの被験者内Pearson相関を計算する。
4. 全被験者の窓を束ねた相関(pooled)と、被験者ごとの相関の中央値の両方を報告する。
5. **trivialベースライン**: (a)常に0を出す予測、(b)train相当(全被験者)のarousal平均を常に
   出す予測、の2つのMAE/相関を併記する(research/CLAUDE.md制約5)。

## 使い方

    .venv/bin/python scripts/evaluate_arousal_from_hr_case.py

データは `data/case/` に `git clone https://gitlab.com/karan-shr/case_dataset.git data/case`
してから `git lfs pull` したものを想定(`data/case/NOTES.md`参照)。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.affect.arousal_from_hr import (  # noqa: E402
    arousal_score_from_rr,
    rpeaks_from_ecg,
)

CASE_ROOT = REPO_ROOT / "data" / "case"
# 注意(2026-08-06訂正): gitlab.com/karan-shr/case_dataset には interpolated/non-interpolated
# のCSV(video列付き)は実体を含まず、READMEのみだった(figshare側の別配布物らしい)。
# 実データがあるのは data/raw/ (タブ区切り、ヘッダなし、video列なし)のみだったため、
# rawデータ + metadata/videos_duration.txt の startVid区間長で代用する。
PHYSIO_DIR = CASE_ROOT / "data" / "raw" / "physiological"
ANNOT_DIR = CASE_ROOT / "data" / "raw" / "annotations"
PHYSIO_COLUMNS = ["daqtime", "ecg", "bvp", "gsr", "rsp", "skt", "emg_zygo", "emg_coru", "emg_trap"]
ANNOT_COLUMNS = ["jstime", "valence", "arousal"]

# metadata/seqs_order.txt: 全30被験者が共通してstartVidを最初に見る(data/case/NOTES.md参照)。
# rawデータにはvideo列が無いため、metadata/videos_duration.txtのstartVid区間長[ms]を使い、
# セッション開始(daqtime=0)からその長さをベースライン区間とみなす。
STARTVID_DURATION_S = 101.5  # metadata/videos_duration.txt: "startVid" 101500 ms
WINDOW_SEC = 10.0
N_SUBJECTS = 30


def _ecg_fs(physio_df: pd.DataFrame) -> float:
    """daqtime[s]列から実効サンプリング周波数を推定する(公称1000Hzだが実測で確認する)。"""
    dt_s = np.median(np.diff(physio_df["daqtime"].to_numpy()))
    return 1.0 / dt_s


def load_subject(subject_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    physio = pd.read_csv(
        PHYSIO_DIR / f"sub{subject_id}_DAQ.txt", sep="\t", header=None, names=PHYSIO_COLUMNS
    )
    annot = pd.read_csv(
        ANNOT_DIR / f"sub{subject_id}_joystick.txt", sep="\t", header=None, names=ANNOT_COLUMNS
    )
    return physio, annot


def window_mean_annotation(annot: pd.DataFrame, window_starts_s: np.ndarray, window_sec: float) -> np.ndarray:
    """CASEのarousalアノテーションを、HR側と同じ非重複窓で平均する。"""
    t_s = annot["jstime"].to_numpy()
    t_s = t_s - t_s[0]  # ECG側と同じ「セッション開始からの経過秒」に揃える
    arousal = annot["arousal"].to_numpy()
    out = np.full(len(window_starts_s), np.nan)
    for i, ws in enumerate(window_starts_s):
        mask = (t_s >= ws) & (t_s < ws + window_sec)
        if mask.sum() > 0:
            out[i] = float(np.mean(arousal[mask]))
    return out


def evaluate_subject(subject_id: int) -> dict | None:
    physio, annot = load_subject(subject_id)
    fs = _ecg_fs(physio)

    daq_t_s = physio["daqtime"].to_numpy()
    daq_t_s = daq_t_s - daq_t_s[0]
    ecg = physio["ecg"].to_numpy()

    r_peak_times_s = rpeaks_from_ecg(ecg, fs)
    if len(r_peak_times_s) < 20:
        return None  # R波検出に失敗した被験者はスキップ(理由をログに残す)

    # rawデータにはvideo列が無いため、セッション開始からstartVidの区間長([0, 101.5s))を
    # ベースラインとみなす(全被験者共通でstartVidが最初に提示されるため。metadata/seqs_order.txt参照)。
    startvid_start, startvid_end = 0.0, STARTVID_DURATION_S

    def is_baseline(window_center_s: float) -> bool:
        return startvid_start <= window_center_s < startvid_end

    result = arousal_score_from_rr(
        r_peak_times_s,
        baseline_mask_fn=is_baseline,
        window_sec=WINDOW_SEC,
        session_start_s=float(daq_t_s[0]),
        session_end_s=float(daq_t_s[-1]),
    )

    annot_windowed = window_mean_annotation(annot, result.window_start_s, WINDOW_SEC)

    valid = ~(np.isnan(result.arousal) | np.isnan(annot_windowed))
    # ベースライン窓自体は「正規化の基準」であり評価対象からは除外する
    # (ベースライン区間はarousalスコアの平均が定義上ゼロ付近になり、相関を人為的に薄める/濃くする)
    non_baseline = np.array(
        [not is_baseline(c) for c in result.window_center_s]
    )
    valid = valid & non_baseline

    if valid.sum() < 5:
        return None

    return {
        "subject_id": subject_id,
        "n_windows": int(valid.sum()),
        "hr_arousal": result.arousal[valid],
        "annot_arousal": annot_windowed[valid],
        "baseline_mean_hr": result.baseline_mean_hr,
        "baseline_std_hr": result.baseline_std_hr,
    }


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    per_subject = []
    skipped = []
    for sid in range(1, N_SUBJECTS + 1):
        csv_path = PHYSIO_DIR / f"sub{sid}_DAQ.txt"
        if not csv_path.exists():
            skipped.append((sid, "no_file"))
            continue
        try:
            r = evaluate_subject(sid)
        except Exception as e:  # noqa: BLE001
            skipped.append((sid, f"error:{e}"))
            continue
        if r is None:
            skipped.append((sid, "insufficient_data"))
            continue
        per_subject.append(r)

    print(f"評価できた被験者数: {len(per_subject)} / {N_SUBJECTS}")
    if skipped:
        print(f"スキップ: {skipped}")

    if not per_subject:
        print("評価できた被験者が0人でした。data/case/ の配置を確認してください。")
        return

    # --- 被験者内相関(各被験者ごとにPearson r、その中央値を報告) ---
    subject_rs = [pearson_r(r["hr_arousal"], r["annot_arousal"]) for r in per_subject]
    subject_rs_valid = [r for r in subject_rs if not np.isnan(r)]

    # --- pooled(全被験者の窓を束ねた)相関 ---
    all_hr_arousal = np.concatenate([r["hr_arousal"] for r in per_subject])
    all_annot_arousal = np.concatenate([r["annot_arousal"] for r in per_subject])
    pooled_r = pearson_r(all_hr_arousal, all_annot_arousal)

    n_total_windows = len(all_hr_arousal)

    # --- trivialベースライン ---
    # (a) 常に0(=正規化後の期待値であるベースラインHRの状態)を出す
    trivial_zero_mae = float(np.mean(np.abs(all_annot_arousal - 0.0)))
    # (b) 全被験者のarousalアノテーション平均を常に出す(leave-one-out的な厳密さは今回は問わない。
    #     「HRを一切見ない場合にどの程度当てられるか」の目安を示すのが目的のため)
    trivial_mean = float(np.mean(all_annot_arousal))
    trivial_mean_mae = float(np.mean(np.abs(all_annot_arousal - trivial_mean)))

    # モデル(HR由来スコア)側のMAE参考値として、スケールを合わせるための単純な線形回帰を1本だけ通す
    # (相関係数は本質的にスケール不変だが、MAEで比較する場合は必要)
    slope, intercept = np.polyfit(all_hr_arousal, all_annot_arousal, 1)
    hr_pred = slope * all_hr_arousal + intercept
    hr_mae = float(np.mean(np.abs(all_annot_arousal - hr_pred)))

    print()
    print("=== 相関(HR由来覚醒度スコア vs CASE連続arousalアノテーション) ===")
    print(f"pooled Pearson r        : {pooled_r:.3f}  (n_windows={n_total_windows})")
    print(
        f"被験者内 Pearson r 中央値: {np.median(subject_rs_valid):.3f}  "
        f"(有効被験者数={len(subject_rs_valid)}/{len(per_subject)})"
    )
    print(f"被験者内 r の分布        : min={np.min(subject_rs_valid):.3f}, max={np.max(subject_rs_valid):.3f}")
    print()
    print("=== trivialベースラインとの比較(MAE, arousalスケール[0.5, 9.5]相当) ===")
    print(f"trivial(常に0)          MAE: {trivial_zero_mae:.3f}")
    print(f"trivial(全体平均{trivial_mean:.3f})  MAE: {trivial_mean_mae:.3f}")
    print(f"HR由来スコア(線形較正後) MAE: {hr_mae:.3f}")
    print()
    print("参考(厳密な再現対象ではない): WESADデータセットでの過去の記憶上の結果は覚醒度R^2≈0.55")
    print(f"本評価でのpooled R^2相当(r^2): {pooled_r**2:.3f}")

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "arousal_from_hr_case_evaluation.md"
    lines = [
        "# CASEデータセットでのHR由来覚醒度スコア評価",
        "",
        f"- 評価できた被験者数: {len(per_subject)} / {N_SUBJECTS}",
        f"- 非重複窓長: {WINDOW_SEC:.0f}秒",
        f"- ベースライン: 各被験者のセッション開始から{STARTVID_DURATION_S:.1f}秒間(startVid区間)",
        "",
        "## 相関",
        "",
        f"- pooled Pearson r: {pooled_r:.3f} (n_windows={n_total_windows})",
        f"- 被験者内 Pearson r 中央値: {np.median(subject_rs_valid):.3f} "
        f"(有効被験者数={len(subject_rs_valid)}/{len(per_subject)})",
        f"- 被験者内 r の範囲: [{np.min(subject_rs_valid):.3f}, {np.max(subject_rs_valid):.3f}]",
        "",
        "## trivialベースラインとの比較(MAE)",
        "",
        "| 手法 | MAE |",
        "|---|---|",
        f"| trivial(常に0) | {trivial_zero_mae:.3f} |",
        f"| trivial(全体平均{trivial_mean:.3f}) | {trivial_mean_mae:.3f} |",
        f"| HR由来覚醒度スコア(線形較正後) | {hr_mae:.3f} |",
        "",
        "## 参考値(厳密な再現対象ではない)",
        "",
        "過去の記憶(MEMORY.md)によればWESADデータセットで覚醒度R^2≈0.55という結果があった。"
        "CASEとWESADは被験者・刺激・アノテーション方式が異なるため直接比較はできないが、"
        f"本評価でのpooled r^2={pooled_r**2:.3f}はオーダー感の参考として記録する。",
        "",
        "## スキップした被験者",
        "",
        f"{skipped}" if skipped else "なし",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
