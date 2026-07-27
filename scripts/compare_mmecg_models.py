"""MMECG(mmWave)向けの全モデル系統を横断比較する。

CNN/Transformer/LSTM/NCP/複素CNN(波形回帰・heatmap回帰)・古典ML(RR Interval直接回帰)は
「R波検出→RR Interval MAE」という共通の物差しで比較できる。一方GNN(+GAN)系統
（214/215、`mmecg_beatgraph[_gan]`）は既知のR波位置を前提に拍単位のPQRST形状を予測する
別種のタスクであり、同じ物差しに乗らないため別表で報告する
（`src/dog_radar_vitals/mmecg_rpeak_evaluation.py`のdocstring参照）。

使い方:
    python scripts/compare_mmecg_models.py --runs runs/xxx_ecg_ecg_cnn1d runs/yyy_rr_interval_ridge ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.mmecg_rpeak_evaluation import evaluate_run_on_test_subjects  # noqa: E402

_SEQ2SEQ_FAMILY_TO_KIND = {
    "mmecg_seq2seq": "waveform",
    "mmecg_rpeak_seq2seq": "heatmap",
    "mmecg_heatmap_gan": "heatmap",  # 生成器はheatmapを出力するため、rpeak_seq2seqと同じ評価が使える
}
_SPATIAL_MODEL_TO_KIND = {
    "ecg_spatial_fusion": "waveform",
    "rpeak_spatial_fusion": "heatmap",
    "ecg_spatial_gnn": "waveform",
    "rpeak_spatial_gnn": "heatmap",
}
_BEATGRAPH_FAMILIES = {"mmecg_beatgraph", "mmecg_beatgraph_gan"}


def _summarize_seq2seq(run_dir: Path, config: dict, model_kind: str) -> dict:
    per_trial = evaluate_run_on_test_subjects(run_dir, config, REPO_ROOT, model_kind)
    f1s = [r["f1"] for r in per_trial]
    rr_maes = [r["rr_mae_ms"] for r in per_trial if r["rr_mae_ms"] is not None]
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "family": config["model"]["family"],
        "n_trials": len(per_trial),
        "f1_mean": float(np.mean(f1s)) if f1s else None,
        "f1_std": float(np.std(f1s)) if f1s else None,
        "rr_mae_ms_mean": float(np.mean(rr_maes)) if rr_maes else None,
        "rr_mae_ms_std": float(np.std(rr_maes)) if rr_maes else None,
        "n_trials_with_rr": len(rr_maes),
    }


def _summarize_classical(run_dir: Path, config: dict) -> dict:
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "family": config["model"]["family"],
        "n_trials": None,
        "f1_mean": None,
        "f1_std": None,
        "rr_mae_ms_mean": test_metrics["mae"],
        "rr_mae_ms_std": None,
        "n_trials_with_rr": None,
    }


def _summarize_beatgraph(run_dir: Path, config: dict) -> dict:
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text())
    return {
        "run": run_dir.name,
        "model": config["model"]["name"],
        "family": config["model"]["family"],
        "time_mae_ms": test_metrics["mae"],
        "amp_mae": test_metrics.get("amp_mae"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="比較したいrun_dirのリスト")
    parser.add_argument("--out", default=None, help="結果jsonの保存先")
    args = parser.parse_args()

    rr_table, beatgraph_table = [], []
    for run_str in args.runs:
        run_dir = Path(run_str)
        config = load_config(run_dir / "config.yaml")
        family = config["model"]["family"]

        if family in _SEQ2SEQ_FAMILY_TO_KIND:
            rr_table.append(_summarize_seq2seq(run_dir, config, _SEQ2SEQ_FAMILY_TO_KIND[family]))
        elif family in ("mmecg_spatial_seq2seq", "mmecg_spatial_heatmap_gan"):
            rr_table.append(_summarize_seq2seq(run_dir, config, _SPATIAL_MODEL_TO_KIND[config["model"]["name"]]))
        elif family == "mmecg_classical_rr":
            rr_table.append(_summarize_classical(run_dir, config))
        elif family in _BEATGRAPH_FAMILIES:
            beatgraph_table.append(_summarize_beatgraph(run_dir, config))
        else:
            print(f"skip: unknown family '{family}' for run {run_dir}")

    print("\n=== RR Interval MAE比較 (CNN/Transformer/LSTM/NCP/複素CNN/古典ML) ===")
    rr_table.sort(key=lambda r: r["rr_mae_ms_mean"] if r["rr_mae_ms_mean"] is not None else float("inf"))
    for r in rr_table:
        f1_str = f"f1={r['f1_mean']:.3f}" if r["f1_mean"] is not None else "f1=N/A"
        rr_str = f"{r['rr_mae_ms_mean']:.1f}ms" if r["rr_mae_ms_mean"] is not None else "N/A"
        print(f"  {r['model']:20s} ({r['family']}): {f1_str}  rr_mae={rr_str}")

    print("\n=== ビートグラフ(PQRST形状)比較 (GNN/GAN、既知R波前提の別タスク) ===")
    for r in beatgraph_table:
        print(f"  {r['model']:20s} ({r['family']}): time_mae={r['time_mae_ms']:.1f}ms  amp_mae={r['amp_mae']:.3f}")

    out_path = Path(args.out) if args.out else REPO_ROOT / "reports" / "mmecg_comparison" / "comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rr_interval": rr_table, "beatgraph": beatgraph_table}, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
