"""エッジ実行性の実測(空間GNN版)。

radarODE-MTL/scripts/benchmark_edge_efficiency.py と同一の測定条件
(CPU 4スレッド、warmup 10 / 計測 50 回、batch=1、4秒窓)で、
本リポジトリ側の RPeakSpatialGNN(posXYZ + k近傍GATConv) を測る。
torch_geometric が本リポジトリの venv にしか入っていないため測定を分けている。
クロスチェックのため、同条件で rpeak_cnn1d(生信号dilated CNN 64ch)も併せて測る。

使い方:
    .venv/bin/python scripts/benchmark_edge_efficiency_gnn.py \
        --out reports/edge_efficiency_spatial_gnn.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

N_WARMUP = 10
N_RUNS = 50
CPU_THREADS = 4


def _gpu_power_sampler(stop_event, samples, interval=0.05):
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=2)
            samples.append(float(out.decode().strip().split("\n")[0]))
        except Exception:
            pass
        time.sleep(interval)


def count_flops(model, inputs):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    counter = FlopCounterMode(display=False)
    try:
        with counter:
            with torch.no_grad():
                model(*inputs)
        return counter.get_total_flops()
    except Exception as exc:  # GATConvのscatter系はカバーされない演算がある
        print(f"  [flops] skipped: {exc}")
        return None


def measure_latency(model, inputs, device, sample_power=False):
    model = model.to(device).eval()
    inputs = [x.to(device) for x in inputs]
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    power_samples, stop = [], threading.Event()
    if sample_power and device.type == "cuda":
        th = threading.Thread(target=_gpu_power_sampler, args=(stop, power_samples))
        th.start()

    times = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    if sample_power and device.type == "cuda":
        stop.set()
        th.join()

    res = {
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_std": float(np.std(times)),
        "latency_ms_p50": float(np.percentile(times, 50)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "realtime_factor_4s_window": float(np.mean(times)) / 4000.0,
    }
    if device.type == "cuda":
        res["peak_gpu_mem_MB"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        if power_samples:
            res["gpu_power_W_mean"] = float(np.mean(power_samples))
            res["gpu_power_W_max"] = float(np.max(power_samples))
    model.to("cpu")
    return res


def benchmark_model(name, model, inputs, note=""):
    n_params = sum(p.numel() for p in model.parameters())
    entry = {
        "name": name, "note": note,
        "params": n_params, "params_M": n_params / 1e6,
        "model_size_MB_fp32": n_params * 4 / 1024 ** 2,
        "input_shapes": [list(x.shape) for x in inputs],
    }
    model = model.eval()
    flops = count_flops(model, inputs)
    if flops is not None:
        entry["flops_per_window"] = flops
        entry["gflops_per_window"] = flops / 1e9
    torch.set_num_threads(CPU_THREADS)
    entry["cpu"] = measure_latency(model, inputs, torch.device("cpu"))
    entry["cpu"]["n_threads"] = CPU_THREADS
    if torch.cuda.is_available():
        entry["gpu"] = measure_latency(model, inputs, torch.device("cuda"), sample_power=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/edge_efficiency_spatial_gnn.json")
    args = ap.parse_args()

    from dog_radar_vitals.models.deep.rpeak_spatial_gnn import RPeakSpatialGNN
    from dog_radar_vitals.models.deep.rpeak_cnn1d import RPeakCNN1D

    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "cpu_threads": CPU_THREADS, "models": [],
    }

    rcg = torch.randn(1, 800, 50)      # (batch, seq_len, n_points)
    pos = torch.randn(1, 50, 3)
    results["models"].append(benchmark_model(
        "spatial_gnn_posxyz", RPeakSpatialGNN(n_points=50, embed_dim=32, k_neighbors=6,
                                              n_gat_layers=2, n_downsample=3),
        [rcg, pos],
        note="点ごと時間conv(8倍縮約)→posXYZのk近傍(k=6)グラフ上でGATConv×2→転置convで800点に復元"))

    results["models"].append(benchmark_model(
        "rpeak_cnn1d_64ch_crosscheck", RPeakCNN1D(in_channels=50, channels=64, n_layers=6,
                                                  kernel_size=15, dropout=0.1),
        [torch.randn(1, 800, 50)],  # 本リポジトリ側は (batch, seq_len, ch) 入力
        note="クロスチェック用(radarODE-MTL側 raw_anchor_cnn_64ch と同一構成)"))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
