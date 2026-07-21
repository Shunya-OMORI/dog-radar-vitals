# runs/

`python -m dog_radar_vitals.train --config ...` を実行すると、実行ごとにここへ
`{タイムスタンプ}_{task}_{model}/` ディレクトリが自動生成される。

各runディレクトリの中身（すべて `.gitignore` により追跡外）:

| ファイル | 内容 |
|---|---|
| `config.yaml` | 実際に使われた設定のスナップショット（`extends` 解決後） |
| `metrics.json` | epochごとのtrain/val損失・MAE |
| `best_model.pt` | val MAE最良時点のモデル重み |
| `test_metrics.json` | `evaluate.py` 実行後に追加されるtest split結果 |

**このディレクトリ自体はgitで追跡しない。** 人間が追跡すべき「どのrunが何を意味するか」は
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) に手動で残す。run_dirのパスをそこに書いておけば、
後から `config.yaml` と `metrics.json` を読み返して再現できる。
