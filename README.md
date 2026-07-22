# dog-radar-vitals

レーダ信号からイヌのバイタルサイン（心拍数・呼吸数）を推定するモデルの実装・実験リポジトリ。

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
│   ├── scenario1.py               # Ahmed et al. (2024) シナリオ1の読み込み
│   ├── windowing.py                # スライディング窓切り出し（deep/classical共通）
│   ├── dataset.py                  # 生の窓をそのまま使う深層モデル向けDataset
│   └── features.py                 # 古典ML向けの手作り特徴量抽出
├── models/
│   ├── deep/
│   │   ├── transformer.py           # VitalsTransformer
│   │   ├── cnn1d.py                 # VitalsCNN1D
│   │   ├── lstm.py                  # VitalsLSTM
│   │   └── registry.py              # 深層モデル名→クラスの一元管理
│   └── classical/
│       └── registry.py              # 古典MLモデル名→scikit-learn Estimatorの一元管理
├── training/
│   ├── deep_trainer.py              # PyTorchの学習/評価ループ
│   ├── classical_trainer.py         # scikit-learnのfit/評価ループ
│   └── schedulers.py                # 学習率スケジューラ（configで明示指定した場合のみ有効）
├── config.py                       # YAML設定の読み込み（extends継承）
├── seeding.py                      # 乱数シード固定
├── reproducibility.py              # gitコミット・パッケージ版の記録
├── train.py                        # 学習CLI（familyでdeep/classicalを振り分け）
└── evaluate.py                     # 評価CLI（同上）
scripts/
├── run_comparison.py                # 複数configの一括学習・評価（実行の責務のみ）
├── make_comparison_report.py        # マニフェストから比較表・グラフを生成（集計の責務のみ）
└── plot_learning_curves.py          # 複数runのval MAE学習曲線を重ねて比較（収束診断用）
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
モデル比較を再度行う前に、この分割の信頼性（leave-few-dogs-out cross-validationの必要性）に
対応することを最優先課題としている。

RR Interval・ECG波形予測、マルチタスク学習、複素領域モデル、超次元コンピューティング、
モデル小型化、健康モニタリングへの拡張予定は [`EXPERIMENTS.md`](EXPERIMENTS.md) の
「今後の拡張予定」を参照。
