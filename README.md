# dog-radar-vitals

レーダ信号からイヌのバイタルサイン（心拍数・呼吸数、および手法検証としてヒトのECG波形）を
推定するモデルの実装・実験リポジトリ。

先行研究調査やテーマ選定の経緯は [`manager-agent/research/dog-mmwave-rri/`](../manager-agent/research/dog-mmwave-rri/) にある。
このリポジトリの責務は**実装側の試行錯誤**に限定する。文献調査・テーマ設計は上記へ。

## 設計方針

コーディングエージェントによる試行錯誤やモデル数の増加があっても、人間が経緯を追えることを優先する。

- **設定はYAMLで宣言し、コードは触らない。** `configs/experiments/*.yaml` が1実験1ファイル。
  差分が小さくgit履歴で追いやすい。番号は連番で増やし、既存ファイルは上書きしない。
- **runs/ は使い捨て、EXPERIMENTS.md が正史。** 学習結果は `runs/{timestamp}/` に自動保存されるが
  `.gitignore` 対象。「何を試して何が分かったか」は [`EXPERIMENTS.md`](EXPERIMENTS.md) に人間が書く。
- **モデルはfamily（deep/classical）ごとにレジストリで一元管理。** 新しい深層モデルは
  `models/deep/` に追加して `models/deep/registry.py` に、新しい古典MLモデルは
  `models/classical/registry.py` に1行足すだけ。`train.py`・`evaluate.py` 側の変更は不要。
- **1ファイル1責務。** データ読み込み・窓切り出し・特徴量抽出・モデル定義・学習ループ・
  再現性記録・比較実行・レポート生成をそれぞれ別ファイルに分離している（下記ディレクトリ構成参照）。
  バージョンごとにファイルを上書きせず、新しいアーキテクチャは新しいファイル名で追加する。
- **train/val/testは犬ID単位で分割。** 同一犬のウィンドウが複数splitに漏れてリークするのを防ぐ
  （`configs/base.yaml` の `data.dogs`）。
- **再現性を最初から作り込む。** 乱数シード固定（`seeding.py`）、gitコミット・主要パッケージ版の
  記録（`reproducibility.py`）を全runで自動的に行う。査読対応で「どのコードのどのバージョンで
  この結果が出たか」を追える設計。

## セットアップ

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

データセットの配置方法は [`data/raw/README.md`](data/raw/README.md) を参照。

## 使い方

```bash
# 1モデルの学習
python -m dog_radar_vitals.train --config configs/experiments/001_transformer_hr.yaml

# 評価（学習で表示されたrun_dirを指定）
python -m dog_radar_vitals.evaluate --run runs/20260722-000000_hr_transformer

# 複数モデルを一括学習・評価して比較する
# （深層モデルはGPU上で逐次、古典MLモデルはCPU上で並列実行）
python scripts/run_comparison.py --all --tag 20260722_baseline
python scripts/make_comparison_report.py --tag 20260722_baseline
# -> reports/20260722_baseline/{table.md,chart.png} が生成される

# テスト
pytest
```

## ディレクトリ構成

```
configs/
├── base.yaml                    # 全実験共通のデフォルト設定（model部分は空、各experimentで完全指定）
└── experiments/                  # 1実験1ファイル。extends: ../base.yaml で継承
data/
├── raw/                          # 生データ（gitignore対象、配置方法はraw/README.md）
└── processed/                    # 前処理キャッシュ（gitignore対象、現在未使用）
src/dog_radar_vitals/
├── data/
│   ├── scenario1.py               # Ahmed et al. (2024) シナリオ1の読み込み（犬HR/BR）
│   ├── windowing.py                # スライディング窓切り出し（犬deep/classical共通、窓->スカラ）
│   ├── dataset.py                  # 生の窓をそのまま使う深層モデル向けDataset（犬HR/BR）
│   ├── features.py                 # 古典ML向けの手作り特徴量抽出（犬HR/BR）
│   ├── schellenberger.py           # Schellenberger et al. (2020) の読み込み（ヒトECG波形）
│   ├── ecg_windowing.py            # 窓切り出し（ヒトECG、窓->同じ長さの波形）
│   └── ecg_dataset.py              # レーダI/Q窓とECG波形窓のDataset（ヒトECG）
├── models/
│   ├── deep/
│   │   ├── transformer.py           # VitalsTransformer（犬HR/BR）
│   │   ├── cnn1d.py                 # VitalsCNN1D（犬HR/BR）
│   │   ├── lstm.py                  # VitalsLSTM（犬HR/BR）
│   │   ├── registry.py              # 犬HR/BR深層モデル名→クラスの一元管理
│   │   ├── ecg_cnn1d.py             # ECGWaveformCNN1D（ヒトECG波形、sequence-to-sequence）
│   │   └── ecg_registry.py          # ECG波形モデル名→クラスの一元管理（registry.pyとは別、入出力の形が違うため）
│   └── classical/
│       └── registry.py              # 古典MLモデル名→scikit-learn Estimatorの一元管理
├── training/
│   ├── deep_trainer.py              # 犬HR/BR: PyTorchの学習/評価ループ（窓->スカラ）
│   ├── classical_trainer.py         # 犬HR/BR: scikit-learnのfit/評価ループ
│   ├── ecg_trainer.py               # ヒトECG波形: 学習/評価ループ（窓->波形、指標は相関係数）
│   └── schedulers.py                # 学習率スケジューラ（configで明示指定した場合のみ有効）
├── config.py                       # YAML設定の読み込み（extends継承）
├── seeding.py                      # 乱数シード固定
├── reproducibility.py              # gitコミット・パッケージ版の記録
├── baselines.py                    # trivial/oracleベースラインの計算（固定分割・CV両方から使う）
├── train.py                        # 学習CLI（familyでdeep/classical/ecg_seq2seqを振り分け）
└── evaluate.py                     # 評価CLI（同上）
scripts/
├── run_comparison.py                # 複数configの一括学習・評価（実行の責務のみ）
├── make_comparison_report.py        # マニフェストから比較表・グラフを生成（集計の責務のみ）
├── plot_learning_curves.py          # 複数runのval MAE学習曲線を重ねて比較（収束診断用）
├── compute_baselines.py             # 固定分割でのtrivial/oracleベースラインCLI
├── diagnose_within_dog_signal.py    # あるrunが個体内の時間変動を追えているか診断
└── run_dog_cross_validation.py      # 犬を入れ替えた5-fold cross-validation（1モデルにつき5run）
runs/                              # 学習結果（gitignore対象、README.md参照）
reports/                           # run_comparisonの結果をまとめた表・グラフ（git管理下）
tests/
EXPERIMENTS.md                     # 人間が維持する実験ログの正史
```

## 現在のタスク・データの制約

現行データセット（Ahmed et al. 2024, *Scientific Data* 11:107、シナリオ1: 麻酔下のイヌ10頭）は
参照値が **1FPSへ平均化された心拍数・呼吸数のスカラ値のみ**であり、拍単位のRR IntervalもECG波形も
含まない。そのため現在実装しているのは**心拍数予測・呼吸数予測の2タスクのみ**。

2026-07-22に深層モデル3種（Transformer/CNN1D/LSTM）と古典ML3種（Ridge/Random Forest/
Gradient Boosting）の初回比較を実施した。結果と考察は [`EXPERIMENTS.md`](EXPERIMENTS.md) の
「本実行から分かったこと」、表とグラフは [`reports/20260722_baseline/`](reports/20260722_baseline/) を参照。

同日、初回比較でTransformer(HR)が未収束だった件を学習率・スケジューラの対照実験で切り分けた
（[`reports/20260722_hr_transformer_ablation/`](reports/20260722_hr_transformer_ablation/)）。
学習率を上げれば古典MLを上回る水準まで改善する一方、**val犬1頭への過適合という、
今回の犬分割（train 7・val 1・test 2の固定1分割）自体に起因するより重大な問題**が見つかった。
詳細は [`EXPERIMENTS.md`](EXPERIMENTS.md) の「HR Transformer対照実験の結果」を参照。

**さらに重要な確認: `scripts/compute_baselines.py` で「レーダを使わず訓練犬の平均値を常に
予測するだけ」のtrivialベースラインを計算したところ、BRタスクは16モデル中1つもこれを
上回れず、HRタスクも明確に上回ったのは2モデルのみだった。** 詳細と今後の方針は
[`EXPERIMENTS.md`](EXPERIMENTS.md) の「【最重要】trivialベースラインとの比較」を参照。

さらに `scripts/diagnose_within_dog_signal.py` で、その2モデルが個体内の時間変動を
実際に追えているか（＝健康モニタリングに使えるか）を検証したところ、**個体内相関はほぼゼロで、
母集団平均への回帰にすぎないことが判明した**（詳細は[`EXPERIMENTS.md`](EXPERIMENTS.md)の
「追試: HRでtrivialを上回った2モデルは健康モニタリングに使えるか」）。現行データ
（麻酔下・1頭1回のスナップショット）は、そもそも個体内の状態変化を検証できる構造になっていない。

**最優先課題だった単一分割の信頼性検証も完了した。** `scripts/run_dog_cross_validation.py` で
犬を入れ替えた5-fold cross-validationを実施したところ、単一分割で「勝者」に見えていた
2モデル（Transformer lr=5e-4、Random Forest）はいずれもfold間で結果が大きく揺れ、
trivialベースラインを安定して上回れないことが判明した（[`reports/20260722_dog_cross_validation/`](reports/20260722_dog_cross_validation/)）。
**単一分割に基づくこれまでの「どのモデルが優れているか」という結論は再現しなかった。**
詳細は [`EXPERIMENTS.md`](EXPERIMENTS.md) の「犬入れ替えcross-validationの結果」を参照。
今後のモデル比較は単一分割ではなくこのCV手順を標準とする。

RR Interval・ECG波形予測、マルチタスク学習、複素領域モデル、超次元コンピューティング、
モデル小型化、健康モニタリングへの拡張予定は [`EXPERIMENTS.md`](EXPERIMENTS.md) の
「今後の拡張予定」を参照。

## ヒトデータでの手法検証（ECG波形推定）

イヌのHR/BR予測が個体内の時間変動を全く追えていなかったため、目標を「他センサの波形推定」に
広げる方向を検証している。イヌには波形の正解データが無いため、まずヒトの公開データセット
（Schellenberger et al. 2020、CW radar + 同期ECG）で手法を確立する
（`configs/experiments/101_ecg_cnn1d_resting.yaml`）。

初回実行でtest_corr=0.524を得た。イヌHR/BRのwithin_dog_corr（実質ゼロ）とは対照的に、
未知の被験者でも一貫して有意な相関が出ており、「レーダから他センサ波形を推定する」方向性
自体は少なくともヒトデータでは成立することを確認した。ただしQRS波の鋭い形状の再現はまだ弱い。
詳細は [`EXPERIMENTS.md`](EXPERIMENTS.md) の「ヒトECG波形推定（手法検証）の結果」、
波形の可視化は [`reports/20260722_ecg_resting/`](reports/20260722_ecg_resting/) を参照。
