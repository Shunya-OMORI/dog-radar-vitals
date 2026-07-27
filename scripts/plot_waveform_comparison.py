"""波形推定モデル(ecg_*系、mmecg_seq2seq/mmecg_spatial_seq2seq family)だけを集めて、
(1) 推定波形と正解波形の重ね描きグラフ、(2) 窓内Pearson相関係数の比較、を出す。

heatmap回帰(rpeak_*)・古典ML(RR Interval直接回帰)・GNN(beatgraph)は対象外
（出力の型が異なり同じ図で比較できないため）。

使い方:
    python scripts/plot_waveform_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects  # noqa: E402
from dog_radar_vitals.data.mmecg_dataset import MMECGWindowDataset  # noqa: E402
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, zscore_channels  # noqa: E402
from dog_radar_vitals.data.mmecg_spatial_dataset import MMECGSpatialWindowDataset  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402
from dog_radar_vitals.training.ecg_trainer import _pearson_corr  # noqa: E402
from dog_radar_vitals.training.spatial_fusion_trainer import build_spatial_model  # noqa: E402

# Okabe-Ito配色。make_comparison_report.py等と同じ規約、カテゴリごとに固定順で割り当てる。
GT_COLOR = "#000000"
PRED_COLOR = "#0072B2"

# (表示名, run_dir, is_complex, is_spatial) のリスト。201-235フェーズで実際に学習した
# 波形推定(ecg_*)モデルすべて。ecg_ncpはepochs=50の224(v2)を代表として使う。
MODELS = [
    ("ecg_cnn1d",          "runs/20260725-011349_ecg_ecg_cnn1d",          False, False),
    ("ecg_transformer",    "runs/20260725-011908_ecg_ecg_transformer",    False, False),
    ("ecg_lstm",           "runs/20260725-015511_ecg_ecg_lstm",           False, False),
    ("ecg_ncp",            "runs/20260727-012755_ecg_ecg_ncp",            False, False),
    ("ecg_conv_ncp",       "runs/20260727-091201_ecg_ecg_conv_ncp",       False, False),
    ("ecg_complex_cnn",    "runs/20260725-020023_ecg_ecg_complex_cnn",    True,  False),
    ("ecg_complex_cnn_v2", "runs/20260725-112418_ecg_ecg_complex_cnn_v2", True,  False),
    ("ecg_conformer",      "runs/20260725-113210_ecg_ecg_conformer",      False, False),
    ("ecg_unet1d",         "runs/20260725-113622_ecg_ecg_unet1d",         False, False),
    ("ecg_spatial_fusion", "runs/20260727-091215_ecg_ecg_spatial_fusion", False, True),
    ("ecg_spatial_gnn",    "runs/20260727-095516_ecg_ecg_spatial_gnn",    False, True),
]

## 注意: トライアルID(.matファイル番号)と被験者IDは別の名前空間である
## (`data/mmecg.py`参照)。例えば「17.mat」は被験者ID=9(訓練用)に属し、被験者ID=17の
## トライアルではない。誤って訓練データを混入させないよう、必ず
## `trial_ids_for_subjects(raw_root, test_subjects)`で実際のtestトライアルIDを取得してから
## 選ぶこと。60番はecg_cnn1dのtrial別相関(0.001〜0.747まで分布)のうち全体平均(0.224)に
## 最も近い「典型的な」testトライアルとして選んだ。
TEST_SUBJECT_IDS = [17, 29, 30]
TEST_TRIAL_ID = 60  # 被験者29に属する、ecg_cnn1dのtrial別相関が全体平均に最も近いtestトライアル
WINDOW_INDICES = [3, 12]  # 同一トライアル内の2つの窓（拍のタイミングが異なる区間）を見る


def _load_model(run_dir: Path, config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family = config["model"]["family"]
    if family == "mmecg_spatial_seq2seq":
        model = build_spatial_model(config["model"]).to(device)
    else:
        model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
        model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


def _get_window(rcg_z: np.ndarray, ecg_z: np.ndarray, start: int, window_len: int, complex_input: bool):
    rcg_win = rcg_z[start : start + window_len]
    ecg_win = ecg_z[start : start + window_len]
    if complex_input:
        rcg_win = analytic_signal_channels(rcg_win)
    return rcg_win, ecg_win


def _full_test_correlation(model, config: dict, device: torch.device, is_complex: bool, is_spatial: bool) -> float:
    """2窓の抜粋ではなく、test被験者の全トライアル・全窓での平均相関（正式なランキング用）。"""
    data_cfg = config["data"]
    raw_root = REPO_ROOT / data_cfg["raw_root"]
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"]["test"])

    if is_spatial:
        ds = MMECGSpatialWindowDataset(raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"])
    else:
        ds = MMECGWindowDataset(raw_root, trial_ids, data_cfg["window_sec"], data_cfg["stride_sec"], complex_input=is_complex)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    total_corr, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if is_spatial:
                rcg, posxyz, y = batch
                pred = model(rcg.to(device), posxyz.to(device))
            else:
                x, y = batch
                pred = model(x.to(device))
            total_corr += _pearson_corr(pred, y.to(device))
            n_batches += 1
    return total_corr / n_batches


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_trials = trial_ids_for_subjects(REPO_ROOT / "data" / "raw", TEST_SUBJECT_IDS)
    assert TEST_TRIAL_ID in test_trials, (
        f"TEST_TRIAL_ID={TEST_TRIAL_ID} はtest被験者{TEST_SUBJECT_IDS}に属さない"
        f"（実際のtestトライアル: {test_trials}）。トライアルID(.matファイル番号)と被験者IDは"
        "別の名前空間なので混同しないこと。"
    )

    rec = load_trial(REPO_ROOT / "data" / "raw", TEST_TRIAL_ID)
    rcg_z = zscore_channels(rec.rcg)
    from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap

    ecg_z = zscore_with_nan_gap(rec.ecg)

    sample_cfg = load_config(REPO_ROOT / MODELS[0][1] / "config.yaml")
    window_len = int(sample_cfg["data"]["window_sec"] * rec.fs)

    results = []
    predictions: dict[str, list[np.ndarray]] = {}
    for name, run_str, is_complex, is_spatial in MODELS:
        run_dir = REPO_ROOT / run_str
        config = load_config(run_dir / "config.yaml")
        model = _load_model(run_dir, config)

        corrs = []
        preds_for_windows = []
        for w_idx in WINDOW_INDICES:
            start = w_idx * window_len
            rcg_win, ecg_win = _get_window(rcg_z, ecg_z, start, window_len, is_complex)

            x = torch.from_numpy(rcg_win).unsqueeze(0)
            x = x.to(torch.complex64) if is_complex else x.float()
            x = x.to(device)
            y = torch.from_numpy(ecg_win).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if is_spatial:
                    posxyz = torch.from_numpy(rec.posxyz).float().unsqueeze(0).to(device)
                    pred = model(x, posxyz)
                else:
                    pred = model(x)

            corrs.append(_pearson_corr(pred, y))
            preds_for_windows.append(pred.cpu().numpy().squeeze(0))

        predictions[name] = preds_for_windows

        # 例示用の2窓だけでなく、test被験者全トライアル・全窓での正式な平均相関も算出する
        # （1トライアルだけだと被験者間のばらつきが非常に大きく、ランキングとして使えないため）。
        full_test_corr = _full_test_correlation(model, config, device, is_complex, is_spatial)

        results.append(
            {
                "model": name,
                "example_trial_corr": float(np.mean(corrs)),
                "example_window_corrs": corrs,
                "full_test_corr": full_test_corr,
            }
        )
        print(
            f"{name:20s} full_test_corr={full_test_corr:.3f}  "
            f"example_trial({TEST_TRIAL_ID})_corr={np.mean(corrs):.3f}  per-window={[f'{c:.3f}' for c in corrs]}"
        )

    results.sort(key=lambda r: r["full_test_corr"], reverse=True)

    # --- 図1: 波形の重ね描き(モデルごとに小さいsubplot、正解=黒、予測=青) ---
    n_models = len(MODELS)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 2.6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    _, ecg_win0 = _get_window(rcg_z, ecg_z, WINDOW_INDICES[0] * window_len, window_len, False)
    t = np.arange(len(ecg_win0)) / rec.fs

    ordered_names = [r["model"] for r in results]
    for i, name in enumerate(ordered_names):
        ax = axes_flat[i]
        ax.plot(t, ecg_win0, color=GT_COLOR, linewidth=1.2, label="Ground truth")
        ax.plot(t, predictions[name][0], color=PRED_COLOR, linewidth=1.2, label="Prediction")
        corr = next(r["full_test_corr"] for r in results if r["model"] == name)
        ax.set_title(f"{name}  (full-test corr={corr:.3f})", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Time [s]", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("ECG (z-scored)", fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(ordered_names), len(axes_flat)):
        axes_flat[j].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Waveform Prediction Comparison (illustrative example: test trial {TEST_TRIAL_ID}, window #{WINDOW_INDICES[0]}; "
        "title corr = full test-set average, not this single window)",
        y=1.06, fontsize=12,
    )
    fig.tight_layout()
    out1 = REPO_ROOT / "reports" / "mmecg_comparison" / "waveform_prediction_grid.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"saved: {out1}")

    # --- 図2: 相関係数の横棒グラフ(ランキング、test被験者全体の正式な平均) ---
    fig2, ax2 = plt.subplots(figsize=(7.5, 0.4 * n_models + 1))
    names_sorted = [r["model"] for r in results]
    corrs_sorted = [r["full_test_corr"] for r in results]
    bars = ax2.barh(names_sorted, corrs_sorted, color=PRED_COLOR, height=0.6)
    for bar, c in zip(bars, corrs_sorted):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{c:.3f}", va="center", fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Pearson correlation over all test-subject windows")
    ax2.set_title("Waveform Models: Correlation Ranking\n(ecg_* family, full test set)")
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    out2 = REPO_ROOT / "reports" / "mmecg_comparison" / "waveform_correlation_ranking.png"
    fig2.savefig(out2, dpi=150)
    print(f"saved: {out2}")

    out_json = REPO_ROOT / "reports" / "mmecg_comparison" / "waveform_prediction_comparison.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_json}")


if __name__ == "__main__":
    main()
