"""比較実験の後処理フェアネス確保用(2026-08-28, ユーザ指摘対応)。

「今の後処理(height閾値など)は現行モデル(config284)に合わせて選んだものであり、
比較対象の代替モデルにそのまま使うのは不公平」という指摘に対応する。
各runについて:
  (a) as-is: config284で使っているheight=0.3・単純find_peaks抽出をそのまま適用
  (b) tuned: valだけを見てheight(・抽出方式)を選び直し、testに適用
の両方を報告する(testを見て選んだ値を報告に使わない、022_height_select_on_val.shと同じ方針)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from dog_radar_vitals.config import load_config
from dog_radar_vitals.data.rpeaks import extract_peaks_dark_refine, extract_peaks_from_box, extract_peaks_from_heatmap
from dog_radar_vitals.mmecg_rpeak_evaluation import evaluate_run_on_test_subjects

EXTRACTORS = {
    "default": extract_peaks_from_heatmap,
    "dark": extract_peaks_dark_refine,
    "box": extract_peaks_from_box,
}


def mean_f1(results: list[dict]) -> float:
    return sum(r["f1"] for r in results) / len(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model_kind", default="heatmap")
    ap.add_argument("--extractor", default="default", choices=list(EXTRACTORS))
    ap.add_argument("--heights", default="0.1,0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--as_is_height", type=float, default=0.3)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    extractor = EXTRACTORS[args.extractor]

    # (a) as-is: config284と同じheight・素の抽出方式で test を評価
    as_is = evaluate_run_on_test_subjects(
        run_dir, config, Path("."), args.model_kind, split="test",
        height=args.as_is_height, peak_extractor=extract_peaks_from_heatmap,
    )
    print(f"[as-is]  height={args.as_is_height} extractor=default  test mean_f1={mean_f1(as_is):.3f}")

    # (b) tuned: val で height を選び、選ばれた height を test に適用(test は選択に使わない)
    heights = [float(h) for h in args.heights.split(",")]
    val_scores = {}
    for h in heights:
        val_results = evaluate_run_on_test_subjects(
            run_dir, config, Path("."), args.model_kind, split="val",
            height=h, peak_extractor=extractor,
        )
        val_scores[h] = mean_f1(val_results)
        print(f"  [val sweep] height={h} extractor={args.extractor}  val mean_f1={val_scores[h]:.3f}")

    best_h = max(val_scores, key=val_scores.get)
    tuned = evaluate_run_on_test_subjects(
        run_dir, config, Path("."), args.model_kind, split="test",
        height=best_h, peak_extractor=extractor,
    )
    print(f"[tuned]  height={best_h}(valで選択) extractor={args.extractor}  test mean_f1={mean_f1(tuned):.3f}")

    out = {
        "as_is": {"height": args.as_is_height, "extractor": "default", "test_mean_f1": mean_f1(as_is)},
        "tuned": {"height": best_h, "extractor": args.extractor, "test_mean_f1": mean_f1(tuned), "val_scores": val_scores},
    }
    (run_dir / "postproc_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
