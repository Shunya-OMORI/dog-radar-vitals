"""複数configを一括学習・評価し、run_dir一覧をマニフェストとして保存する。

GPUは1枚のため深層モデル(family: deep)は逐次実行する。古典MLモデル(family: classical)は
CPU上で学習が完結するため、ProcessPoolExecutorで並列実行する。

実行結果の集計・表・グラフ生成は責務を分離し `scripts/make_comparison_report.py` が担う。
このスクリプトは「学習を回してrun_dirを確定させる」ことだけに責任を持つ。

使い方:
    python scripts/run_comparison.py --tag 20260722_baseline \
        --configs configs/experiments/001_transformer_hr.yaml configs/experiments/002_transformer_br.yaml ...
    python scripts/run_comparison.py --tag 20260722_baseline --all
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.evaluate import evaluate_run  # noqa: E402
from dog_radar_vitals.train import make_run_dir, train_from_config  # noqa: E402


def _run_one(config_path: str, tag: str) -> dict:
    config = load_config(config_path)
    run_dir = make_run_dir(config, tag=tag)
    print(f"[{config_path}] run_dir: {run_dir}")
    train_from_config(config, run_dir)
    test_metrics = evaluate_run(run_dir)
    return {
        "config_path": config_path,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "task": config["task"],
        "family": config["model"]["family"],
        "model": config["model"]["name"],
        "test_mae": test_metrics["mae"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None, help="実行するconfigファイルのパス群")
    parser.add_argument("--all", action="store_true", help="configs/experiments/*.yaml を全て実行する")
    parser.add_argument("--tag", required=True, help="マニフェストファイル名・run_dirサフィックスに使うタグ")
    parser.add_argument("--classical-workers", type=int, default=4, help="古典MLモデルを並列実行するプロセス数")
    args = parser.parse_args()

    if args.all:
        config_paths = sorted(str(p) for p in (REPO_ROOT / "configs" / "experiments").glob("*.yaml"))
    elif args.configs:
        config_paths = args.configs
    else:
        parser.error("--configs か --all のいずれかを指定してください")
        return

    deep_paths = [p for p in config_paths if load_config(p)["model"]["family"] == "deep"]
    classical_paths = [p for p in config_paths if load_config(p)["model"]["family"] == "classical"]

    results = []

    print(f"=== deep models: {len(deep_paths)}件を逐次実行(GPU) ===")
    for p in deep_paths:
        results.append(_run_one(p, args.tag))

    print(f"=== classical models: {len(classical_paths)}件を並列実行(CPU, workers={args.classical_workers}) ===")
    if classical_paths:
        # 先に深層モデルがCUDAを初期化しているため、fork継承だとCUDA再初期化エラーになる。
        # 古典MLはCPUのみで完結するのでspawnで独立プロセスにする。
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.classical_workers, mp_context=ctx) as executor:
            futures = [executor.submit(_run_one, p, args.tag) for p in classical_paths]
            for future in futures:
                results.append(future.result())

    manifest_dir = REPO_ROOT / "runs" / "_comparisons"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.tag}.json"
    manifest_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
