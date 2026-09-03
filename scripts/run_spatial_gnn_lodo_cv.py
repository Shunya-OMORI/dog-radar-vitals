"""採用構成(config284, 空間GNN)を11名全員でLeave-One-Subject-Out交差検証にかける。

なぜ必要か(2026-08-21):
    我々の報告値(F1 0.749 / RR間隔MAE 8.41ms)は train7/val1/test3 の単一分割で得た値である。
    一方 radarODE / radarODE-MTL は論文本文で「被験者単位の11分割交差検証」と述べている。
    分割が違うままでは，どちらが良いかを論じられない。**相手の実装を再現するのではなく，
    相手の評価プロトコルをこちらが採用する**ことで対等な比較を作る(進捗方針 5節)。

**単一変数(R4)**: モデル構成・損失・前処理は config284 から一切変えない。変えるのは被験者分割だけ。

各 fold: 9名train・1名val・1名test。11名なので11 fold で全員が一度ずつ test に回る。
`run_radarode_sceg_lodo_cv.py` と同じ組み方にしてある。

チェックポイントの選び方:
    全 fold で **epoch_004.pt に固定**する。採用構成が epoch4 だったのに合わせるためであり，
    fold ごとに test 上で最良 epoch を選ぶと選択バイアスが入る。
    (限界: epoch4 という選択自体は単一分割の test 上で事後に決めたものである。この点は報告に併記する)

学習のみを行い，HRV指標(F1/RR間隔MAE/RMSSD)の評価は
`radarODE-MTL/scripts/evaluate_spatial_gnn_hrv.py --test_subjects <fold の test 被験者>`
に任せる。過去の報告値を出したのと同一の評価経路を使うため(R3)。

使い方:
    .venv/bin/python scripts/run_spatial_gnn_lodo_cv.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dog_radar_vitals.config import load_config  # noqa: E402
from dog_radar_vitals.train import make_run_dir, train_from_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/284_mmecg_rpeak_spatial_gnn_neurokit_sigma15_full.yaml"
ALL_SUBJECTS = [1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30]


def main() -> None:
    # 2026-08-24: 50点の混ぜ方を変えた対照(290/291)も同じ手続きで回せるようにする。
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tag", default="lodo284", help="run 名と出力 json の接頭辞")
    args = ap.parse_args()
    output_path = REPO_ROOT / f"reports/{args.tag}_cv_runs.json"
    base_config = load_config(args.config)
    results = []

    for i, test_subject in enumerate(ALL_SUBJECTS):
        remaining = [s for s in ALL_SUBJECTS if s != test_subject]
        val_subject = remaining[i % len(remaining)]
        train_subjects = [s for s in remaining if s != val_subject]

        config = json.loads(json.dumps(base_config))  # deep copy
        config["data"]["subjects"] = {
            "train": train_subjects,
            "val": [val_subject],
            "test": [test_subject],
        }

        tag = f"{args.tag}_fold{i}_test{test_subject}"
        run_dir = make_run_dir(config, tag=tag)
        print(f"\n=== fold {i}: test_subject={test_subject} val_subject={val_subject} ===", flush=True)
        print(f"train_subjects={train_subjects}", flush=True)
        train_from_config(config, run_dir)

        ckpt = run_dir / "epoch_004.pt"
        print(f"fold {i} done. checkpoint exists: {ckpt.exists()} -> {ckpt}", flush=True)

        results.append(
            {
                "fold": i,
                "test_subject": test_subject,
                "val_subject": val_subject,
                "train_subjects": train_subjects,
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt),
                "checkpoint_exists": ckpt.exists(),
            }
        )
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n=== LOSO 学習完了: {len(results)} fold ===")
    print(f"run 一覧: {output_path}")
    print("次は evaluate_spatial_gnn_hrv.py を fold ごとに --test_subjects つきで回す(キュー014)")


if __name__ == "__main__":
    main()
