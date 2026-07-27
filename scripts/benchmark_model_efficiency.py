"""ecg_registryの各モデルについて、パラメータ数・推定FLOPs・CPU推論レイテンシを測定する。

NCP(本命1)を「精度だけでなく低容量・低消費電力」という観点で正式に評価するための
ベンチマークCLI。「Efficient Edge-AI Models for Robust ECG Abnormality Detection on
Resource-Constrained Hardware」(PMC, 2024)がSTM32マイクロコントローラ上でLTC/CfCの
RAM・フラッシュ・消費電力・推論レイテンシを測定した手法に倣い、GPUではなくCPU上での
バッチ1件あたり推論レイテンシを組込み展開のプロキシとして測定する
（`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「NCP/LTCの効率性評価」参照）。

使い方:
    python scripts/benchmark_model_efficiency.py --out reports/mmecg_comparison/efficiency_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ptflops import get_model_complexity_info  # noqa: E402

from dog_radar_vitals.data.mmecg import load_trial  # noqa: E402
from dog_radar_vitals.models.deep.ecg_registry import build_ecg_model  # noqa: E402
from dog_radar_vitals.training.spatial_fusion_trainer import build_spatial_model  # noqa: E402

SEQ_LEN = 800  # window_sec(4) * fs(200Hz)、201-215のMMECG configと同一
IN_CHANNELS = 50
N_LATENCY_RUNS = 50
N_WARMUP_RUNS = 10

# (登録名, モデル構築kwargs, 複素入力か) のリスト。201-215で実際に使ったkwargsと揃える。
_BENCHMARK_TARGETS: list[tuple[str, dict, bool]] = [
    ("ecg_cnn1d", dict(in_channels=IN_CHANNELS, channels=64, n_layers=6, kernel_size=15, dropout=0.1), False),
    ("rpeak_cnn1d", dict(in_channels=IN_CHANNELS, channels=64, n_layers=6, kernel_size=15, dropout=0.1), False),
    ("ecg_transformer", dict(in_channels=IN_CHANNELS, d_model=128, n_heads=4, n_layers=4, dim_feedforward=256, dropout=0.1), False),
    ("rpeak_transformer", dict(in_channels=IN_CHANNELS, d_model=128, n_heads=4, n_layers=4, dim_feedforward=256, dropout=0.1), False),
    ("ecg_lstm", dict(in_channels=IN_CHANNELS, hidden_size=64, n_layers=2, dropout=0.1), False),
    ("rpeak_lstm", dict(in_channels=IN_CHANNELS, hidden_size=64, n_layers=2, dropout=0.1), False),
    ("ecg_ncp", dict(in_channels=IN_CHANNELS, units=64, output_dim=1), False),
    ("rpeak_ncp", dict(in_channels=IN_CHANNELS, units=64, output_dim=1), False),
    ("ecg_complex_cnn", dict(in_channels=IN_CHANNELS, channels=32, n_layers=6, kernel_size=15), True),
    ("rpeak_complex_cnn", dict(in_channels=IN_CHANNELS, channels=32, n_layers=6, kernel_size=15), True),
    ("ecg_complex_cnn_v2", dict(in_channels=IN_CHANNELS, channels=32, n_blocks=6, kernel_size=15), True),
    ("rpeak_complex_cnn_v2", dict(in_channels=IN_CHANNELS, channels=32, n_blocks=6, kernel_size=15), True),
    (
        "ecg_conformer",
        dict(in_channels=IN_CHANNELS, d_model=128, conv_kernel_size=15, conv_layers=3, n_heads=4, n_transformer_layers=4, dim_feedforward=256, dropout=0.1),
        False,
    ),
    (
        "rpeak_conformer",
        dict(in_channels=IN_CHANNELS, d_model=128, conv_kernel_size=15, conv_layers=3, n_heads=4, n_transformer_layers=4, dim_feedforward=256, dropout=0.1),
        False,
    ),
    ("ecg_unet1d", dict(in_channels=IN_CHANNELS, base_channels=32, depth=3, kernel_size=9), False),
    ("rpeak_unet1d", dict(in_channels=IN_CHANNELS, base_channels=32, depth=3, kernel_size=9), False),
    (
        "ecg_conv_ncp",
        dict(in_channels=IN_CHANNELS, conv_channels=32, conv_kernel_size=15, conv_layers=3, ncp_units=64, ncp_output_dim=8, n_ncp_layers=2, mixed_memory=True),
        False,
    ),
    (
        "rpeak_conv_ncp",
        dict(in_channels=IN_CHANNELS, conv_channels=32, conv_kernel_size=15, conv_layers=3, ncp_units=64, ncp_output_dim=8, n_ncp_layers=2, mixed_memory=True),
        False,
    ),
]

# 2引数forward(rcg, posxyz)を取る空間モデル。上記の_BENCHMARK_TARGETSとは入力形が異なるため別扱い。
_SPATIAL_BENCHMARK_TARGETS: list[dict] = [
    dict(family="mmecg_spatial_seq2seq", name="ecg_spatial_fusion", n_points=IN_CHANNELS, embed_dim=32, n_heads=4, n_attn_layers=2),
    dict(family="mmecg_spatial_seq2seq", name="rpeak_spatial_fusion", n_points=IN_CHANNELS, embed_dim=32, n_heads=4, n_attn_layers=2),
    dict(family="mmecg_spatial_seq2seq", name="ecg_spatial_gnn", n_points=IN_CHANNELS, embed_dim=32, k_neighbors=6, n_gat_layers=2),
    dict(family="mmecg_spatial_seq2seq", name="rpeak_spatial_gnn", n_points=IN_CHANNELS, embed_dim=32, k_neighbors=6, n_gat_layers=2),
]


def _measure_cpu_latency_ms(model: torch.nn.Module, complex_input: bool) -> float:
    model = model.to("cpu").eval()
    dtype = torch.complex64 if complex_input else torch.float32
    x = torch.randn(1, SEQ_LEN, IN_CHANNELS, dtype=dtype)

    with torch.no_grad():
        for _ in range(N_WARMUP_RUNS):
            model(x)
        start = time.perf_counter()
        for _ in range(N_LATENCY_RUNS):
            model(x)
        elapsed = time.perf_counter() - start
    return elapsed / N_LATENCY_RUNS * 1000.0


def benchmark_model(name: str, kwargs: dict, complex_input: bool) -> dict:
    model = build_ecg_model(name, **kwargs)
    n_params = sum(p.numel() for p in model.parameters())

    dtype = torch.complex64 if complex_input else torch.float32
    macs, _ = get_model_complexity_info(
        model,
        (SEQ_LEN, IN_CHANNELS),
        as_strings=False,
        print_per_layer_stat=False,
        input_constructor=lambda shape: {"x": torch.randn(1, *shape, dtype=dtype)},
    )

    latency_ms = _measure_cpu_latency_ms(model, complex_input)

    return {
        "model": name,
        "n_params": int(n_params),
        "macs": int(macs) if macs is not None else None,
        "cpu_latency_ms": latency_ms,
    }


def _measure_cpu_latency_ms_spatial(model: torch.nn.Module, posxyz: torch.Tensor) -> float:
    model = model.to("cpu").eval()
    x = torch.randn(1, SEQ_LEN, IN_CHANNELS)

    with torch.no_grad():
        for _ in range(N_WARMUP_RUNS):
            model(x, posxyz)
        start = time.perf_counter()
        for _ in range(N_LATENCY_RUNS):
            model(x, posxyz)
        elapsed = time.perf_counter() - start
    return elapsed / N_LATENCY_RUNS * 1000.0


def benchmark_spatial_model(model_cfg: dict) -> dict:
    """2引数forward(rcg, posxyz)を取る空間モデル用。posXYZは実データ(トライアル1)を使う。"""
    model = build_spatial_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())

    rec = load_trial(REPO_ROOT / "data" / "raw", 1)
    posxyz = torch.from_numpy(rec.posxyz).float().unsqueeze(0)

    macs, _ = get_model_complexity_info(
        model,
        (SEQ_LEN, IN_CHANNELS),
        as_strings=False,
        print_per_layer_stat=False,
        input_constructor=lambda shape: {"rcg": torch.randn(1, *shape), "posxyz": posxyz},
    )

    latency_ms = _measure_cpu_latency_ms_spatial(model, posxyz)

    return {
        "model": model_cfg["name"],
        "n_params": int(n_params),
        "macs": int(macs) if macs is not None else None,
        "cpu_latency_ms": latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = []
    for name, kwargs, complex_input in _BENCHMARK_TARGETS:
        r = benchmark_model(name, kwargs, complex_input)
        results.append(r)
        macs_str = f"{r['macs']/1e6:.1f}M MACs" if r["macs"] is not None else "MACs=N/A"
        print(f"{r['model']:20s} params={r['n_params']:>9,}  {macs_str:>14s}  cpu_latency={r['cpu_latency_ms']:.3f}ms")

    for model_cfg in _SPATIAL_BENCHMARK_TARGETS:
        r = benchmark_spatial_model(model_cfg)
        results.append(r)
        macs_str = f"{r['macs']/1e6:.1f}M MACs" if r["macs"] is not None else "MACs=N/A"
        print(f"{r['model']:20s} params={r['n_params']:>9,}  {macs_str:>14s}  cpu_latency={r['cpu_latency_ms']:.3f}ms")

    out_path = Path(args.out) if args.out else REPO_ROOT / "reports" / "mmecg_comparison" / "efficiency_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
