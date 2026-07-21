このリポジトリは、レーダ→イヌのバイタルサイン推定モデルの実装・実験を行う場所である。
コーディングエージェントによる試行錯誤が前提のため、人間が後から経緯を追えることを最優先する。

## 新しい実験を追加するとき

- ハイパーパラメータやデータ分割をコードに埋め込まない。`configs/experiments/*.yaml` に
  新しいファイルを作り、`extends: ../base.yaml` で差分だけ書く。
- 学習を実行したら、成否によらず [`EXPERIMENTS.md`](EXPERIMENTS.md) に1行追記する
  （run_dir・設定・狙い・結果・次の一手）。これを省略すると、runs/ はgitignore対象なので
  実験の存在自体が失われる。

## 新しいモデルを追加するとき

- `src/dog_radar_vitals/models/` に新しいモジュールを追加し、`models/registry.py` の
  `MODEL_REGISTRY` に1行足す。`train.py`・`evaluate.py` は変更しない。

## データを扱うとき

- train/val/testの分割は犬ID単位で行う（`configs/base.yaml` の `data.dogs`）。
  同一犬のウィンドウを複数splitに跨がせない。
- `data/raw/` と `runs/` はgitignore対象。新しいデータセットを追加する場合は
  `data/raw/README.md` に取得元と構造を追記する。

## やらないこと

- 既存のconfigファイルを書き換えて過去の実験を上書きしない。新しい設定は新しいファイルとして追加する。
- `runs/` の中身をgit管理下に置かない（チェックポイントは大きく、再現は `config.yaml` から可能なため）。
- RR Interval・ECG波形予測は、現行データセットに正解ラベルがないため実装しない
  （経緯は [`EXPERIMENTS.md`](EXPERIMENTS.md) の「今後の拡張予定」、データの制約は
  [`data/raw/README.md`](data/raw/README.md) を参照）。
