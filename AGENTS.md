このリポジトリは、レーダ→イヌのバイタルサイン推定モデルの実装・実験を行う場所である
（加えて、手法検証のためのヒトデータでのECG波形推定も含む）。
コーディングエージェントによる試行錯誤が前提のため、人間が後から経緯を追えることを最優先する。

## 新しい実験を追加するとき

- ハイパーパラメータやデータ分割をコードに埋め込まない。`configs/experiments/*.yaml` に
  新しいファイルを作り、`extends: ../base.yaml` で差分だけ書く。
- 学習を実行したら、成否によらず [`EXPERIMENTS.md`](EXPERIMENTS.md) に1行追記する
  （run_dir・設定・狙い・結果・次の一手）。これを省略すると、runs/ はgitignore対象なので
  実験の存在自体が失われる。

## 新しいモデルを追加するとき

- 犬HR/BR用の深層モデル(PyTorch)なら `src/dog_radar_vitals/models/deep/` に新しいモジュールを
  追加し、`models/deep/registry.py` の `DEEP_MODEL_REGISTRY` に1行足す。
- 犬HR/BR用の古典MLモデル(scikit-learn)なら `models/classical/registry.py` の
  `CLASSICAL_MODEL_REGISTRY` に1行足す。
- ヒトECG関連（波形回帰・R波heatmap回帰とも）の深層モデルなら `models/deep/ecg_registry.py` の
  `ECG_MODEL_REGISTRY` に1行足す（`registry.py`とは別。窓->スカラ と 窓->系列 で
  呼び出し規約が異なるためレジストリを分けている。波形回帰とheatmap回帰は呼び出し規約が
  同じなので同じレジストリに同居させている）。
- 既存の4つのfamily（deep/classical/ecg_seq2seq/rpeak_seq2seq）の範囲内でモデルを追加するだけなら
  `train.py`・`evaluate.py`・`training/*.py` は変更しない（`config["model"]["family"]` で
  自動的に振り分けられる）。**新しいfamily自体を追加する場合**（例: 犬でも波形推定を
  始める、複素領域モデル用に別の入出力形が要る等）は、対応する`training/*_trainer.py`を
  新設した上で`train.py`/`evaluate.py`の`_TRAIN_FNS`/`_EVAL_FNS`辞書に1行足す
  （これは既存family内のモデル追加とは別の変更単位であり、README.mdのディレクトリ構成も
  合わせて更新する）。
- 1ファイル1責務を保つ。既存のモデルファイルに新モデルを追記せず、新しいファイル名で追加する。

## 比較実験を回すとき

- 単発の学習・評価は `train.py`/`evaluate.py`、複数モデルの一括比較は
  `scripts/run_comparison.py`（実行）→ `scripts/make_comparison_report.py`（集計・グラフ化）
  の順で使う。この2つは責務を分けているので、学習をやり直さずレポートだけ再生成できる。
- 深層モデルはGPUが1枚のため逐次実行、古典MLモデルはCPU並列で実行される
  （`run_comparison.py` が自動で振り分ける）。
- **犬HR/BRのモデル比較は、単一のtrain/val/test分割だけで結論を出さない。**
  2026-07-22のcross-validationで、単一分割での「勝者」がfoldを入れ替えると再現しない
  ことが実証された（[`EXPERIMENTS.md`](EXPERIMENTS.md)「犬入れ替えcross-validationの結果」）。
  「モデルAがモデルBより優れている」と主張する際は `scripts/run_dog_cross_validation.py`
  でfold平均・foldごとのtrivial比較を必ず添える。単一分割の結果は動作確認以上の意味を持たない。
- **fold平均・fold数(n_folds)は`--n-folds 10`（10頭でLeave-One-Dog-Out）を既定とする。**
  さらに`scripts/test_cv_significance.py`で対応のある統計検定（符号検定・Wilcoxon符号順位検定）
  を必ず添える。連続量の回帰誤差にはチャンスレベルが無いため、fold平均を目で比べるだけでは
  「有意な差」なのか「偶然」なのか判断できない（2026-07-23、`EXPERIMENTS.md`
  「Leave-One-Dog-Out CVと統計検定」参照）。

## データを扱うとき

- train/val/testの分割は犬ID単位で行う（`configs/base.yaml` の `data.dogs`）。
  同一犬のウィンドウを複数splitに跨がせない。
- `data/raw/` と `runs/` はgitignore対象。新しいデータセットを追加する場合は
  `data/raw/README.md` に取得元と構造を追記する。

## やらないこと

- 既存のconfigファイルを書き換えて過去の実験を上書きしない。新しい設定は新しいファイルとして追加する。
- `runs/` の中身をgit管理下に置かない（チェックポイントは大きく、再現は `config.yaml` から可能なため）。
- **イヌのRR Interval・ECG波形予測**は、現行の犬データセットに正解ラベルがないため実装しない
  （経緯は [`EXPERIMENTS.md`](EXPERIMENTS.md) の「今後の拡張予定」、データの制約は
  [`data/raw/README.md`](data/raw/README.md) を参照）。**ヒトのECG波形推定・R波検出**
  （手法検証、Schellenbergerデータセット、family="ecg_seq2seq"/"rpeak_seq2seq"）は
  これとは別物で実装済み。混同しない。

## configの番号帯

- `001`〜: 犬HR/BR予測（deep/classical）
- `101`〜: ヒトデータでの手法検証（ecg_seq2seq, rpeak_seq2seq等）
新しいfamilyや対象（例: 犬でのECG波形推定、複素領域モデル等）を追加する際は、
100番台ずつ新しい帯を割り当て、このAGENTS.mdに追記する。
