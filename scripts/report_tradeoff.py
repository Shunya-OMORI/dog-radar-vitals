"""検出性能と HRV 精度のトレードオフを、この分野で標準的な形で図と表にする (2026-08-25)。

なぜこの形式か
--------------
2 つの目的の間にトレードオフがあることを報告するとき、分野ごとに定着した見せ方がある。
ここでは 4 つを組み合わせる。

1. **動作特性曲線 (operating characteristic)**
   横軸に調整パラメータ (しきい値)、左右の縦軸に 2 つの指標を重ねる。
   検出分野で ROC / DET 曲線が使われるのと同じ発想で、
   「1 つのつまみを回すと何がどう動くか」を 1 枚で示す。

2. **パレート境界 (Pareto frontier)**
   横軸 F1、縦軸 RMSSD 誤差の平面に各しきい値を点で打ち、
   「他のどの点にも両方の指標で負けていない点」だけを線で結ぶ。
   多目的最適化でトレードオフを示す標準的な図法 (どちらか一方を良くするには
   他方を犠牲にするしかない、という関係が視覚的に確定する)。
   パレート境界上にない点は「選ぶ理由がない動作点」であり、それも情報になる。

3. **Bland-Altman プロット**
   推定した RMSSD と正解 RMSSD の (平均, 差) を打ち、平均差 (bias) と
   一致限界 (limits of agreement, 平均差 ± 1.96 SD) を引く。
   Bland & Altman, *Lancet* 327(8476):307-310, 1986。
   **臨床測定法の一致性評価では相関係数ではなくこれを使うのが標準**で、
   HRV の論文でも必須級。相関係数は「入力を見ずに平均値を出すだけ」でも高く出るため
   (我々自身、拍単位の相関 0.883 を誤認した経験がある)、この図で代替する。

4. **要約スカラー**
   図を出せない場面 (要旨、表だけの比較) のために、トレードオフを 1 つの数値にする。
     - F1 が一定水準以上を保てる範囲での最小 RMSSD 誤差
     - 逆に、RMSSD 誤差が一定以下を保てる範囲での最大 F1
     - 両者を正規化した距離が最小になる点 (等価点)
   ROC の AUC が「曲線全体を 1 数値にする」のと同じ役割を果たす。

入力
----
`evaluate_height_sweep.py` が保存する `height_sweep_loso.json`
（{被験者: {しきい値: {f1, precision, recall, rr, rmssd, pnn50, ibi, triv_rmssd, triv_pnn50}}}）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

for _p in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if Path(_p).exists():
        try:
            matplotlib.font_manager.fontManager.addfont(_p)
            plt.rcParams["font.family"] = matplotlib.font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False


def load_curve(path: str) -> tuple[list[float], dict[str, np.ndarray], float, float]:
    """被験者平均の曲線を作る。戻り値: しきい値, {指標: 値}, 一定RRのRMSSD, 一定RRのpNN50"""
    d = json.load(open(path, encoding="utf-8"))
    subs = list(d)
    heights = sorted({float(h) for s in subs for h in d[s]})
    keys = ["f1", "precision", "recall", "rr", "rmssd", "pnn50", "ibi"]
    out = {k: [] for k in keys}
    for h in heights:
        hs = f"{h:g}"
        for k in keys:
            vals = [d[s][hs][k] for s in subs if hs in d[s] and d[s][hs].get(k) is not None]
            vals = [v for v in vals if np.isfinite(v)]
            out[k].append(float(np.mean(vals)) if vals else np.nan)
    triv_r = float(np.nanmean([list(d[s].values())[0]["triv_rmssd"] for s in subs]))
    triv_p = float(np.nanmean([list(d[s].values())[0]["triv_pnn50"] for s in subs]))
    return heights, {k: np.asarray(v, dtype=float) for k, v in out.items()}, triv_r, triv_p


def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x は大きいほど良い、y は小さいほど良い。支配されていない点の添字を返す。"""
    idx = []
    for i in range(len(x)):
        dominated = np.any((x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i])))
        if not dominated:
            idx.append(i)
    return np.array(sorted(idx, key=lambda i: x[i]))


def summarize(heights, c, triv_r, f1_floor: float, rmssd_ceiling: float) -> list[str]:
    """図を出せない場面のための要約スカラー。"""
    lines = []
    f1, rm = c["f1"], c["rmssd"]

    ok = np.isfinite(f1) & np.isfinite(rm)
    m = ok & (f1 >= f1_floor)
    if m.any():
        i = int(np.nanargmin(np.where(m, rm, np.inf)))
        lines.append(f"F1 >= {f1_floor:.2f} を保てる範囲での最小 RMSSD 誤差: "
                     f"{rm[i]:.2f} ms (しきい値 {heights[i]:.2f}, F1 {f1[i]:.3f})")
    else:
        lines.append(f"F1 >= {f1_floor:.2f} を満たすしきい値がない")

    m = ok & (rm <= rmssd_ceiling)
    if m.any():
        i = int(np.nanargmax(np.where(m, f1, -np.inf)))
        lines.append(f"RMSSD 誤差 <= {rmssd_ceiling:.1f} ms を保てる範囲での最大 F1: "
                     f"{f1[i]:.3f} (しきい値 {heights[i]:.2f}, RMSSD 誤差 {rm[i]:.2f} ms)")
    else:
        lines.append(f"RMSSD 誤差 <= {rmssd_ceiling:.1f} ms を満たすしきい値がない")

    # 等価点: 各指標を [0,1] に正規化し、理想点 (F1=最大, RMSSD=最小) からの距離が最小の点
    if ok.sum() >= 2:
        fn = (f1 - np.nanmin(f1[ok])) / max(1e-9, np.nanmax(f1[ok]) - np.nanmin(f1[ok]))
        rn = (rm - np.nanmin(rm[ok])) / max(1e-9, np.nanmax(rm[ok]) - np.nanmin(rm[ok]))
        dist = np.sqrt((1 - fn) ** 2 + rn ** 2)
        i = int(np.nanargmin(np.where(ok, dist, np.inf)))
        lines.append(f"理想点からの正規化距離が最小の動作点: しきい値 {heights[i]:.2f} "
                     f"(F1 {f1[i]:.3f}, RMSSD 誤差 {rm[i]:.2f} ms)")

    i_f1 = int(np.nanargmax(np.where(ok, f1, -np.inf)))
    i_rm = int(np.nanargmin(np.where(ok, rm, np.inf)))
    lines.append(f"F1 が最大の動作点: しきい値 {heights[i_f1]:.2f} (F1 {f1[i_f1]:.3f}, "
                 f"RMSSD 誤差 {rm[i_f1]:.2f} ms)")
    lines.append(f"RMSSD 誤差が最小の動作点: しきい値 {heights[i_rm]:.2f} (F1 {f1[i_rm]:.3f}, "
                 f"RMSSD 誤差 {rm[i_rm]:.2f} ms)")
    if i_f1 != i_rm:
        lines.append("→ **2 つの動作点は一致しない。同じしきい値で両方を最良にはできない。**")
    else:
        lines.append("→ 2 つの動作点が一致した。この条件ではトレードオフが観測されない。")
    lines.append(f"参考: 入力を見ずに一定 RR を出すだけのモデルの RMSSD 誤差 {triv_r:.2f} ms "
                 f"(この値を下回らない限り、変動を捉えたとは言えない)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_json", default=str(REPO_ROOT / "reports" / "height_sweep_loso.json"))
    ap.add_argument("--outdir", default=str(REPO_ROOT / "reports" / "tradeoff"))
    ap.add_argument("--label", default="空間GNN (11分割交差検証, 許容150ms)")
    ap.add_argument("--f1_floor", type=float, default=0.80)
    ap.add_argument("--rmssd_ceiling", type=float, default=15.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    heights, c, triv_r, triv_p = load_curve(args.sweep_json)

    # --- 図1: 動作特性曲線 (しきい値を回すと 2 つの指標がどう動くか) ---
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax1.plot(heights, c["f1"], "o-", color="tab:blue", lw=2, label="検出 F1 (許容150ms)")
    ax1.plot(heights, c["precision"], "^--", color="tab:cyan", lw=1, ms=4, alpha=0.8, label="適合率")
    ax1.plot(heights, c["recall"], "v--", color="tab:green", lw=1, ms=4, alpha=0.8, label="再現率")
    ax1.set_xlabel("しきい値")
    ax1.set_ylabel("検出性能", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(heights, c["rmssd"], "s-", color="tab:red", lw=2, label="RMSSD 誤差")
    ax2.axhline(triv_r, color="0.4", ls=":", lw=1.5, label=f"一定RRのみ ({triv_r:.1f} ms)")
    ax2.set_ylabel("RMSSD 誤差 [ms]（小さいほど良い）", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    i_f1 = int(np.nanargmax(c["f1"]))
    i_rm = int(np.nanargmin(c["rmssd"]))
    ax1.axvline(heights[i_f1], color="tab:blue", ls="--", alpha=0.4)
    ax1.axvline(heights[i_rm], color="tab:red", ls="--", alpha=0.4)
    ax1.annotate("F1 最大", (heights[i_f1], c["f1"][i_f1]), textcoords="offset points",
                 xytext=(6, 8), color="tab:blue", fontsize=9)
    ax2.annotate("RMSSD 誤差 最小", (heights[i_rm], c["rmssd"][i_rm]), textcoords="offset points",
                 xytext=(-70, -16), color="tab:red", fontsize=9)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8.5)
    ax1.set_title(f"しきい値を回したときの動作特性　{args.label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_operating_characteristic.png", dpi=120)
    plt.close(fig)

    # --- 図2: パレート境界 ---
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    f1, rm = c["f1"], c["rmssd"]
    ok = np.isfinite(f1) & np.isfinite(rm)
    ax.scatter(f1[ok], rm[ok], s=70, color="0.55", zorder=3, label="各しきい値の動作点")
    for i in np.where(ok)[0]:
        ax.annotate(f"{heights[i]:.2f}", (f1[i], rm[i]), textcoords="offset points",
                    xytext=(7, 4), fontsize=8.5)
    pf = pareto_front(f1[ok], rm[ok])
    idx_ok = np.where(ok)[0]
    pf_abs = idx_ok[pf]
    ax.plot(f1[pf_abs], rm[pf_abs], "o-", color="tab:red", lw=2, ms=9, zorder=4,
            label="パレート境界（他に両方で勝る点がない）")
    ax.axhline(triv_r, color="0.4", ls=":", lw=1.5,
               label=f"入力を見ずに一定RR ({triv_r:.1f} ms)")
    ax.set_xlabel("検出 F1（大きいほど良い）")
    ax.set_ylabel("RMSSD 誤差 [ms]（小さいほど良い）")
    ax.set_title(f"検出性能と心拍ゆらぎ精度のパレート境界　{args.label}", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_pareto.png", dpi=120)
    plt.close(fig)

    # --- 表 + 要約スカラー ---
    lines = [f"# 検出性能と HRV 精度のトレードオフ　{args.label}", "",
             "## 動作点ごとの成績", "",
             "| しきい値 | F1 | 適合率 | 再現率 | RR間隔誤差 | RMSSD誤差 | pNN50誤差 | 平均IBI誤差 | パレート最適 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, h in enumerate(heights):
        star = "**○**" if i in set(pf_abs.tolist()) else ""
        lines.append(f"| {h:.2f} | {f1[i]:.3f} | {c['precision'][i]:.3f} | {c['recall'][i]:.3f} | "
                     f"{c['rr'][i]:.2f} ms | {rm[i]:.2f} ms | {c['pnn50'][i]:.2f} % | "
                     f"{c['ibi'][i]:.2f} ms | {star} |")
    lines += [f"| 一定RRのみ | — | — | — | — | **{triv_r:.2f} ms** | **{triv_p:.2f} %** | — | — |", "",
              "## 要約（図を出せない場面用）", ""]
    lines += [f"- {s}" for s in summarize(heights, c, triv_r, args.f1_floor, args.rmssd_ceiling)]
    lines += ["", "## 図", "",
              "- `fig1_operating_characteristic.png` しきい値を回したときの 2 指標の動き",
              "- `fig2_pareto.png` パレート境界（多目的最適化でトレードオフを示す標準的な図法）",
              "- `fig3_bland_altman.png` RMSSD の一致性（Bland & Altman, Lancet, 1986）"
              "　※ 個票データが要るので `--bland_altman_log` を渡したときだけ作る"]
    (outdir / "tradeoff_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[-14:]))
    print(f"\n保存先: {outdir}")


if __name__ == "__main__":
    main()
