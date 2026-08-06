"""heatmapキーポイント検出(102_rpeak_cnn1d)の「教師信号」と「推論時の極大点検出」の対応を図示する。

進捗報告書の説明(`build_peak_heatmap`でR波位置に立てたガウシアンを教師信号にし、
`extract_peaks_from_heatmap`で推論時にheatmapの極大点を拾ってR波位置に戻す、という2段構え)は
文章だけでは伝わりにくいため、実データ・実モデルで3段構成の図にする。

**用語について**: このタスクは回帰でも(単純な)分類でもなく、物体検出でいうキーポイント検出と
同じ定式化(進捗報告書III-2節)。損失はBCEであり、連続値を当てる「回帰」ではなく、各時刻の
「R波中心らしさ」をソフトラベル([0,1]のガウシアン)で教師する密な分類に近い。「heatmap回帰」
という呼び方は不正確なので本スクリプトでは使わない。

1段目: 正解ECG波形(参照信号)と正解R波位置(縦線) — モデルはこれを直接は見ない
2段目: レーダI/Q(実部、モデルへの実際の入力)と正解R波位置(縦線)
3段目: 教師信号(build_peak_heatmapで作ったガウシアンheatmap) — 学習時にモデルはこれをBCE損失で学習する
4段目: モデルの予測heatmap + extract_peaks_from_heatmapで検出した極大点(推論時の動作)

使い方:
    .venv/bin/python scripts/plot_heatmap_teacher_vs_inference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK JP"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap  # noqa: E402
from dog_radar_vitals.data.rpeaks import build_peak_heatmap, detect_r_peaks, extract_peaks_from_heatmap  # noqa: E402
from dog_radar_vitals.data.schellenberger import load_recording  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402

# Okabe-Ito配色。plot_waveform_comparison.py等、他の図と規約を揃える。
RADAR_COLOR = "#999999"
TRUE_PEAK_COLOR = "#000000"
TEACHER_COLOR = "#009E73"   # 緑: 教師信号(学習時にモデルへ与える正解)
PRED_COLOR = "#0072B2"      # 青: モデルの予測
DETECTED_COLOR = "#D55E00"  # 朱: 推論時に検出した極大点

CONFIG_PATH = REPO_ROOT / "configs/experiments/102_rpeak_cnn1d_resting.yaml"
RUN_DIR = REPO_ROOT / "runs/20260723-003215_ecg_rpeak_cnn1d"
TEST_SUBJECT = "GDN0009"
WINDOW_SEC = 4
# 「概念を伝える」ための図であることを踏まえ、正解4個・検出4個が全一致(誤差最大6ms)する
# 窓を全走査で見つけて採用(ユーザ指示: 今回はイメージ優先で良い窓を選んでよい)。
# モデルの実力(F1 0.32〜0.57)そのものはこの図の主題ではなく、他の節の数値表で報告済み。
PLOT_START_SEC = 111.0


def main() -> None:
    config = load_config(str(CONFIG_PATH))
    data_cfg = config["data"]
    fs = None

    rec = load_recording(REPO_ROOT / data_cfg["raw_root"], TEST_SUBJECT, data_cfg["scenario"])
    fs = rec.fs
    start = int(PLOT_START_SEC * fs)
    length = int(WINDOW_SEC * fs)
    end = start + length

    # iter_peak_windows(rpeak_windowing.py)と同じ前処理(z-score)でモデル入力を作る
    radar_iq_full = np.stack([zscore_with_nan_gap(rec.radar_i), zscore_with_nan_gap(rec.radar_q)], axis=-1)
    radar_iq = radar_iq_full[start:end]
    t = np.arange(length) / fs

    # --- 教師信号(build_peak_heatmapがそのまま学習ターゲットとして使う値) ---
    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    true_peaks_full = detect_r_peaks(ecg_filled, fs)
    teacher_heatmap_full = build_peak_heatmap(true_peaks_full, length=len(ecg_filled), fs=fs)
    teacher_heatmap = teacher_heatmap_full[start:end]
    true_peaks_in_window = true_peaks_full[(true_peaks_full >= start) & (true_peaks_full < end)] - start

    # --- モデルの予測 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(RUN_DIR / "best_model.pt", map_location=device))
    model.eval()

    x = torch.from_numpy(radar_iq).float().unsqueeze(0).to(device)  # (1, T, 2)
    with torch.no_grad():
        pred_heatmap = model(x).squeeze(0).squeeze(-1).cpu().numpy()
    pred_heatmap = np.clip(pred_heatmap, 0.0, 1.0)

    # --- 推論時の極大点検出(build_peak_heatmapの逆操作) ---
    detected_peaks = extract_peaks_from_heatmap(pred_heatmap, fs, height=0.3)

    ecg_window = ecg_filled[start:end]

    # --- 図 ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, ecg_window, color="#000000", lw=1.0, label="正解ECG波形(参照信号)")
    for i, p in enumerate(true_peaks_in_window):
        ax.axvline(t[p], color=TRUE_PEAK_COLOR, lw=0.8, ls="--", alpha=0.7, label="正解R波位置" if i == 0 else None)
    ax.set_ylabel("ECG振幅")
    ax.set_title(f"参照ECG波形 ({TEST_SUBJECT}, {data_cfg['scenario']}, t={PLOT_START_SEC:.0f}-{PLOT_START_SEC+WINDOW_SEC:.0f}s)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    ax.plot(t, radar_iq[:, 0], color=RADAR_COLOR, lw=1.0, label="レーダI/Q(実部、モデルへの入力)")
    for i, p in enumerate(true_peaks_in_window):
        ax.axvline(t[p], color=TRUE_PEAK_COLOR, lw=0.8, ls="--", alpha=0.7, label="正解R波位置" if i == 0 else None)
    ax.set_ylabel("レーダ振幅")
    ax.set_title("入力: レーダI/Q(モデルはECGではなくこちらを見る)")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.fill_between(t, teacher_heatmap, color=TEACHER_COLOR, alpha=0.3)
    ax.plot(t, teacher_heatmap, color=TEACHER_COLOR, lw=1.5, label="教師信号(正解R波にガウシアンを立てたheatmap)")
    for p in true_peaks_in_window:
        ax.axvline(t[p], color=TRUE_PEAK_COLOR, lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("heatmap値")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("学習時: モデルはこの教師信号(build_peak_heatmap)をBCE損失で学習する", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[3]
    ax.fill_between(t, pred_heatmap, color=PRED_COLOR, alpha=0.3)
    ax.plot(t, pred_heatmap, color=PRED_COLOR, lw=1.5, label="モデルの予測heatmap")
    ax.axhline(0.3, color="gray", lw=0.8, ls=":", label="検出しきい値(height=0.3)")
    if len(detected_peaks) > 0:
        ax.scatter(
            t[detected_peaks], pred_heatmap[detected_peaks],
            color=DETECTED_COLOR, zorder=5, s=60, marker="v",
            label="推論時に検出した極大点(extract_peaks_from_heatmap)",
        )
    for p in true_peaks_in_window:
        ax.axvline(t[p], color=TRUE_PEAK_COLOR, lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("heatmap値")
    ax.set_xlabel("時刻 [s]")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("推論時: 予測heatmapの極大点を拾ってR波位置に戻す(build_peak_heatmapの逆操作)", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("heatmapキーポイント検出(102_rpeak_cnn1d): 教師信号と推論時の極大点検出の対応", fontsize=12)
    fig.tight_layout()

    out_path = REPO_ROOT / "reports/mmecg_comparison/heatmap_teacher_vs_inference.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")

    n_true = len(true_peaks_in_window)
    n_detected = len(detected_peaks)
    print(f"この窓内: 正解R波 {n_true}個, 検出された極大点 {n_detected}個")


if __name__ == "__main__":
    main()
