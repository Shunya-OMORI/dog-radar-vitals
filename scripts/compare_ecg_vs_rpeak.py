"""波形回帰(101 ecg_cnn1d)とheatmap回帰(102 rpeak_cnn1d)を、RR Interval精度という
共通の物差しで比較する。

`ecg_trainer.py`の評価指標(波形の相関係数)と`rpeak_trainer.py`の評価指標(heatmapの相関係数)は
互いに比較できない。ここでは両モデルの予測から改めてR波を検出し、真のR波（Schellenbergerの
ECGから検出）とのタイミング一致度・RR Interval誤差を揃えて比較する。

使い方:
    python scripts/compare_ecg_vs_rpeak.py --ecg-run runs/xxx_ecg_ecg_cnn1d --rpeak-run runs/yyy_ecg_rpeak_cnn1d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap  # noqa: E402
from dog_radar_vitals.data.rpeaks import (  # noqa: E402
    detect_r_peaks,
    extract_peaks_from_heatmap,
    match_peaks,
    matched_rr_interval_mae_ms,
)
from dog_radar_vitals.data.schellenberger import load_recording  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402


def _load_model(run_dir: Path, config: dict) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


def _predict_full_recording(model: torch.nn.Module, radar_iq: np.ndarray, window_len: int) -> np.ndarray:
    """非重複窓でレコーディング全体を推論し、連結して1本の系列に戻す。"""
    device = next(model.parameters()).device
    n_steps = radar_iq.shape[0]
    n_windows = n_steps // window_len

    outputs = []
    with torch.no_grad():
        for i in range(n_windows):
            window = radar_iq[i * window_len : (i + 1) * window_len]
            x = torch.from_numpy(window).float().unsqueeze(0).to(device)
            pred = model(x).cpu().numpy().squeeze(0)
            outputs.append(pred)
    return np.concatenate(outputs) if outputs else np.array([])


def evaluate_subject(subject_id: str, raw_root: Path, scenario: str, window_sec: float, ecg_model, rpeak_model) -> dict:
    rec = load_recording(raw_root, subject_id, scenario)
    radar_iq = np.stack([zscore_with_nan_gap(rec.radar_i), zscore_with_nan_gap(rec.radar_q)], axis=-1)
    window_len = int(window_sec * rec.fs)

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    true_peaks = detect_r_peaks(ecg_filled, rec.fs)

    pred_ecg = _predict_full_recording(ecg_model, radar_iq, window_len)
    pred_ecg_peaks = detect_r_peaks(pred_ecg, rec.fs)
    # 予測波形は録音全体の非重複窓の連結であり真値と同じ長さになるはずだが、
    # 端数窓を切り捨てているぶん短くなりうるので、真のピークもその長さに合わせて絞る。
    true_peaks_for_ecg = true_peaks[true_peaks < len(pred_ecg)]
    ecg_match = match_peaks(true_peaks_for_ecg, pred_ecg_peaks, rec.fs, tolerance_ms=50)
    ecg_rr_mae = matched_rr_interval_mae_ms(ecg_match, rec.fs)

    pred_heatmap = _predict_full_recording(rpeak_model, radar_iq, window_len)
    pred_rpeak_peaks = extract_peaks_from_heatmap(pred_heatmap, rec.fs, height=0.3)
    true_peaks_for_rpeak = true_peaks[true_peaks < len(pred_heatmap)]
    rpeak_match = match_peaks(true_peaks_for_rpeak, pred_rpeak_peaks, rec.fs, tolerance_ms=50)
    rpeak_rr_mae = matched_rr_interval_mae_ms(rpeak_match, rec.fs)

    return {
        "subject": subject_id,
        "n_true_peaks": len(true_peaks),
        "ecg_cnn1d": {
            "precision": ecg_match["precision"],
            "recall": ecg_match["recall"],
            "f1": ecg_match["f1"],
            "rr_mae_ms": ecg_rr_mae,
        },
        "rpeak_cnn1d": {
            "precision": rpeak_match["precision"],
            "recall": rpeak_match["recall"],
            "f1": rpeak_match["f1"],
            "rr_mae_ms": rpeak_rr_mae,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecg-run", required=True, help="101(ecg_cnn1d)のrun_dir")
    parser.add_argument("--rpeak-run", required=True, help="102(rpeak_cnn1d)のrun_dir")
    parser.add_argument("--out", default=None, help="結果jsonの保存先")
    args = parser.parse_args()

    ecg_run_dir = Path(args.ecg_run)
    rpeak_run_dir = Path(args.rpeak_run)
    ecg_config = load_config(ecg_run_dir / "config.yaml")
    rpeak_config = load_config(rpeak_run_dir / "config.yaml")

    raw_root = REPO_ROOT / ecg_config["data"]["raw_root"]
    scenario = ecg_config["data"]["scenario"]
    window_sec = ecg_config["data"]["window_sec"]
    test_subjects = ecg_config["data"]["subjects"]["test"]

    ecg_model = _load_model(ecg_run_dir, ecg_config)
    rpeak_model = _load_model(rpeak_run_dir, rpeak_config)

    results = []
    for subject_id in test_subjects:
        r = evaluate_subject(subject_id, raw_root, scenario, window_sec, ecg_model, rpeak_model)
        results.append(r)
        print(f"=== {subject_id} (n_true_peaks={r['n_true_peaks']}) ===")
        for model_name in ["ecg_cnn1d", "rpeak_cnn1d"]:
            m = r[model_name]
            rr_mae_str = f"{m['rr_mae_ms']:.1f}ms" if m["rr_mae_ms"] is not None else "N/A"
            print(f"  {model_name}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} rr_mae={rr_mae_str}")

    out_path = Path(args.out) if args.out else REPO_ROOT / "reports" / "20260723_ecg_vs_rpeak" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
