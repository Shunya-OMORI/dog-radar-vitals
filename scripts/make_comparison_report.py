"""run_comparison.pyが生成したマニフェストを読み、比較表とグラフを reports/ に出力する。

学習の実行(run_comparison.py)とは責務を分離している。マニフェストさえあれば、
再学習せずに何度でも表・グラフを作り直せる。

使い方:
    python scripts/make_comparison_report.py --tag 20260722_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito配色（色弱者にも判別しやすい定番の定性配色）。モデルファミリーごとに固定の色を割り当てる。
FAMILY_COLORS = {
    "deep": "#0072B2",
    "classical": "#E69F00",
}


def load_manifest(tag: str) -> list[dict]:
    manifest_path = REPO_ROOT / "runs" / "_comparisons" / f"{tag}.json"
    return json.loads(manifest_path.read_text())


def make_table(results: list[dict]) -> str:
    header = "| task | model | family | test_mae | run_dir |\n|---|---|---|---|---|\n"
    rows = []
    for r in sorted(results, key=lambda r: (r["task"], r["test_mae"])):
        rows.append(f"| {r['task']} | {r['model']} | {r['family']} | {r['test_mae']:.3f} | `{r['run_dir']}` |")
    return header + "\n".join(rows) + "\n"


def make_chart(results: list[dict], out_path: Path) -> None:
    tasks = sorted({r["task"] for r in results})
    # 環境によってはCJKフォントが未インストールのため、グラフ内の文字列は英語で統一する。
    task_labels = {"hr": "Heart Rate (HR)", "br": "Breathing Rate (BR)"}

    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5), squeeze=False)
    axes = axes[0]

    for ax, task in zip(axes, tasks):
        task_results = sorted((r for r in results if r["task"] == task), key=lambda r: r["test_mae"])
        models = [r["model"] for r in task_results]
        maes = [r["test_mae"] for r in task_results]
        colors = [FAMILY_COLORS[r["family"]] for r in task_results]

        bars = ax.barh(models, maes, color=colors, height=0.6)
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_width() + max(maes) * 0.01, bar.get_y() + bar.get_height() / 2, f"{mae:.2f}", va="center", fontsize=9)

        ax.set_xlabel("Test MAE")
        ax.set_title(task_labels.get(task, task))
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(legend_handles, FAMILY_COLORS.keys(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    results = load_manifest(args.tag)

    out_dir = REPO_ROOT / "reports" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    table_md = make_table(results)
    (out_dir / "table.md").write_text(table_md)

    make_chart(results, out_dir / "chart.png")

    print(table_md)
    print(f"table: {out_dir / 'table.md'}")
    print(f"chart: {out_dir / 'chart.png'}")


if __name__ == "__main__":
    main()
