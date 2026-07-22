"""101(波形回帰)・102(heatmap回帰)を、共通のR波検出・RR Interval精度で評価する共有ロジック。

`ecg_trainer.py`の評価指標(波形の相関係数)と`rpeak_trainer.py`の評価指標(heatmapの相関係数)は
互いに比較できないため、両モデルの予測から改めてR波を検出し、真のR波との一致度・RR Interval
誤差という共通の物差しに変換する。`scripts/compare_ecg_vs_rpeak.py`（単発run同士の比較）と
`scripts/run_ecg_cross_validation.py`（cross-validation）の両方から使う。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dog_radar_vitals.data.ecg_windowing import zscore_with_nan_gap
from dog_radar_vitals.data.rpeaks import (
    detect_r_peaks,
    extract_peaks_from_heatmap,
    match_peaks,
    matched_rr_interval_mae_ms,
)
from dog_radar_vitals.data.schellenberger import load_recording
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model

ModelKind = Literal["waveform", "heatmap"]


def load_ecg_model(run_dir: Path, config: dict) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("family", "name")}
    model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


def predict_full_recording(model: torch.nn.Module, radar_iq: np.ndarray, window_len: int) -> np.ndarray:
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


def evaluate_peak_detection(
    model: torch.nn.Module,
    model_kind: ModelKind,
    subject_id: str,
    raw_root: Path,
    scenario: str,
    window_sec: float,
) -> dict:
    """あるモデルの予測からR波を再検出し、真のR波との一致度・RR Interval誤差を返す。

    model_kind="waveform": 101系。予測波形に`detect_r_peaks`を適用する。
    model_kind="heatmap": 102系。予測heatmapに`extract_peaks_from_heatmap`を適用する。
    """
    rec = load_recording(raw_root, subject_id, scenario)
    radar_iq = np.stack([zscore_with_nan_gap(rec.radar_i), zscore_with_nan_gap(rec.radar_q)], axis=-1)
    window_len = int(window_sec * rec.fs)

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    true_peaks = detect_r_peaks(ecg_filled, rec.fs)

    pred = predict_full_recording(model, radar_iq, window_len)
    if model_kind == "waveform":
        pred_peaks = detect_r_peaks(pred, rec.fs)
    elif model_kind == "heatmap":
        pred_peaks = extract_peaks_from_heatmap(pred, rec.fs, height=0.3)
    else:
        raise ValueError(f"unknown model_kind '{model_kind}'")

    # 予測系列は非重複窓の連結で真値よりわずかに短くなりうるため、真のピークもその長さに合わせる。
    true_peaks_trimmed = true_peaks[true_peaks < len(pred)]
    match = match_peaks(true_peaks_trimmed, pred_peaks, rec.fs, tolerance_ms=50)
    rr_mae = matched_rr_interval_mae_ms(match, rec.fs)

    return {
        "subject": subject_id,
        "n_true_peaks": int(len(true_peaks_trimmed)),
        "precision": match["precision"],
        "recall": match["recall"],
        "f1": match["f1"],
        "rr_mae_ms": rr_mae,
    }
