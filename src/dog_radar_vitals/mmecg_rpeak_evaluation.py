"""101/102用の`rpeak_evaluation.py`のMMECG版。波形回帰・heatmap回帰・分類MLの3系統すべてを
「R波検出→RR Interval MAE」という共通の物差しで評価する（横断比較スクリプト用）。

`mmecg_beatgraph`系統（214/215）はここでは扱わない。あちらは既知のR波位置を前提に
拍単位のPQRST形状を予測するタスクであり、「R波をゼロから検出してRR Intervalを求める」
という本モジュールの前提と噛み合わない（R波検出精度に依存しない別種のタスク）。
`training/beatgraph_trainer.py`の`time_mae_ms`（PQRST各点のタイミング誤差）で別途評価する。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch

from dog_radar_vitals.data.mmecg import load_trial, trial_ids_for_subjects
from dog_radar_vitals.data.mmecg_windowing import analytic_signal_channels, zscore_channels
from dog_radar_vitals.data.rpeaks import detect_r_peaks, extract_peaks_from_heatmap, match_peaks, matched_rr_interval_mae_ms
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model

ModelKind = Literal["waveform", "heatmap"]


_NON_MODEL_KEYS = ("family", "name", "discriminator", "adv_weight")  # mmecg_heatmap_ganの生成器以外のキー
SPATIAL_FAMILIES = {"mmecg_spatial_seq2seq", "mmecg_spatial_heatmap_gan"}  # forward(rcg, posxyz)の2引数family


def load_mmecg_model(run_dir: Path, config: dict) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["model"]["family"] in SPATIAL_FAMILIES:
        from dog_radar_vitals.training.spatial_fusion_trainer import build_spatial_model

        model = build_spatial_model(config["model"]).to(device)
    else:
        model_kwargs = {k: v for k, v in config["model"].items() if k not in _NON_MODEL_KEYS}
        model = build_ecg_model(config["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model.eval()
    return model


def _predict_full_recording(
    model: torch.nn.Module, rcg_input: np.ndarray, window_len: int, posxyz: np.ndarray | None = None
) -> np.ndarray:
    device = next(model.parameters()).device
    n_steps = rcg_input.shape[0]
    n_windows = n_steps // window_len

    posxyz_tensor = None
    if posxyz is not None:
        posxyz_tensor = torch.from_numpy(posxyz).float().unsqueeze(0).to(device)

    outputs = []
    with torch.no_grad():
        for i in range(n_windows):
            window = rcg_input[i * window_len : (i + 1) * window_len]
            x_tensor = torch.from_numpy(window).unsqueeze(0).to(device)
            x_tensor = x_tensor.to(torch.complex64) if np.iscomplexobj(window) else x_tensor.float()
            pred = model(x_tensor, posxyz_tensor) if posxyz_tensor is not None else model(x_tensor)
            pred = pred.cpu().numpy().squeeze(0)
            outputs.append(pred)
    return np.concatenate(outputs) if outputs else np.array([])


def evaluate_trial_peak_detection(
    model: torch.nn.Module,
    model_kind: ModelKind,
    trial_id: int,
    raw_root: Path,
    window_sec: float,
    complex_input: bool = False,
    use_posxyz: bool = False,
    height: float = 0.3,
    peak_extractor=None,
) -> dict:
    """peak_extractor: (heatmap, fs, height) -> peak_indices を受け取る差し替え可能な後処理。
    既定はextract_peaks_from_heatmap(単純find_peaks)。2026-08-28: DARK系のTaylor展開デコード
    や矩形マスク版(extract_peaks_from_box)など、モデルは変えず後処理だけ差し替えた比較に使う。
    """
    rec = load_trial(raw_root, trial_id)
    rcg_z = zscore_channels(rec.rcg)
    rcg_input = analytic_signal_channels(rcg_z) if complex_input else rcg_z
    window_len = int(window_sec * rec.fs)

    ecg_filled = np.nan_to_num(rec.ecg, nan=float(np.nanmean(rec.ecg)))
    true_peaks = detect_r_peaks(ecg_filled, rec.fs)

    pred = _predict_full_recording(model, rcg_input, window_len, posxyz=rec.posxyz if use_posxyz else None)
    if model_kind == "waveform":
        pred_peaks = detect_r_peaks(pred, rec.fs)
    elif model_kind == "heatmap":
        extractor = peak_extractor or extract_peaks_from_heatmap
        pred_peaks = extractor(pred, rec.fs, height)
    else:
        raise ValueError(f"unknown model_kind '{model_kind}'")

    true_peaks_trimmed = true_peaks[true_peaks < len(pred)]
    match = match_peaks(true_peaks_trimmed, pred_peaks, rec.fs, tolerance_ms=50)
    rr_mae = matched_rr_interval_mae_ms(match, rec.fs)

    return {
        "trial_id": trial_id,
        "n_true_peaks": int(len(true_peaks_trimmed)),
        "precision": match["precision"],
        "recall": match["recall"],
        "f1": match["f1"],
        "rr_mae_ms": rr_mae,
    }


def evaluate_run_on_test_subjects(
    run_dir: Path,
    config: dict,
    repo_root: Path,
    model_kind: ModelKind,
    split: str = "test",
    height: float = 0.3,
    peak_extractor=None,
) -> list[dict]:
    """runのsplit(既定test)被験者すべてのトライアルについてevaluate_trial_peak_detectionを実行する。

    split="val": 2026-08-28、後処理(height等)をtestを見ずにvalだけで選ぶための追加。
    """
    data_cfg = config["data"]
    raw_root = repo_root / data_cfg["raw_root"]
    trial_ids = trial_ids_for_subjects(raw_root, data_cfg["subjects"][split])

    model = load_mmecg_model(run_dir, config)
    complex_input = data_cfg.get("complex_input", False)
    use_posxyz = config["model"]["family"] in SPATIAL_FAMILIES
    return [
        evaluate_trial_peak_detection(
            model, model_kind, trial_id, raw_root, data_cfg["window_sec"], complex_input, use_posxyz,
            height=height, peak_extractor=peak_extractor,
        )
        for trial_id in trial_ids
    ]
