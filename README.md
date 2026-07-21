# dog-radar-vitals

レーダ信号からイヌのバイタルサイン（心拍数・呼吸数）を推定するモデルの実装・実験リポジトリ。

先行研究調査やテーマ選定の経緯は [`manager-agent/research/dog-mmwave-rri/`](../manager-agent/research/dog-mmwave-rri/) にある。
このリポジトリの責務は**実装側の試行錯誤**に限定する。文献調査・テーマ設計は上記へ。

## 設計方針

コーディングエージェントによる試行錯誤やモデル数の増加があっても、人間が経緯を追えることを優先する。

- **設定はYAMLで宣言し、コードは触らない。** `configs/experiments/*.yaml` が1実験1ファイル。
  差分が小さくgit履歴で追いやすい。
- **runs/ は使い捨て、EXPERIMENTS.md が正史。** 学習結果は `runs/{timestamp}/` に自動保存されるが
  `.gitignore` 対象。「何を試して何が分かったか」は [`EXPERIMENTS.md`](EXPERIMENTS.md) に人間が書く。
- **モデルはレジストリ経由で一元管理。** 新モデルは `src/dog_radar_vitals/models/` に追加し
  `models/registry.py` に1行足すだけ。`train.py`・`evaluate.py` 側の変更は不要。
- **train/val/testは犬ID単位で分割。** 同一犬のウィンドウが複数splitに漏れてリークするのを防ぐ
  （`configs/base.yaml` の `data.dogs`）。

## セットアップ

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

データセットの配置方法は [`data/raw/README.md`](data/raw/README.md) を参照。

## 使い方

```bash
# 学習（心拍数予測のベースライン）
python -m dog_radar_vitals.train --config configs/experiments/001_transformer_hr.yaml

# 評価（学習で表示されたrun_dirを指定）
python -m dog_radar_vitals.evaluate --run runs/20260722-000000_hr_transformer

# テスト
pytest
```

## ディレクトリ構成

```
configs/
├── base.yaml                 # 全実験共通のデフォルト設定
└── experiments/               # 1実験1ファイル。extends: ../base.yaml で継承
data/
├── raw/                       # 生データ（gitignore対象、配置方法はraw/README.md）
└── processed/                 # 前処理キャッシュ（gitignore対象、現在未使用）
src/dog_radar_vitals/
├── data/
│   ├── scenario1.py           # Ahmed et al. (2024) シナリオ1の読み込み
│   └── dataset.py             # スライディング窓Dataset、犬単位split
├── models/
│   ├── transformer.py         # VitalsTransformer（素朴なTransformer Encoder）
│   └── registry.py            # モデル名→クラスの一元管理
├── config.py                  # YAML設定の読み込み（extends継承）
├── train.py                   # 学習ループ、runs/への記録
└── evaluate.py                # test splitでの評価
runs/                          # 学習結果（gitignore対象、README.md参照）
tests/
EXPERIMENTS.md                 # 人間が維持する実験ログの正史
```

## 現在のタスク・データの制約

現行データセット（Ahmed et al. 2024, *Scientific Data* 11:107、シナリオ1: 麻酔下のイヌ10頭）は
参照値が **1FPSへ平均化された心拍数・呼吸数のスカラ値のみ**であり、拍単位のRR IntervalもECG波形も
含まない。そのため現在実装しているのは**心拍数予測・呼吸数予測の2タスクのみ**。

RR Interval・ECG波形予測、マルチタスク学習、複素領域モデル、超次元コンピューティング、
モデル小型化、健康モニタリングへの拡張予定は [`EXPERIMENTS.md`](EXPERIMENTS.md) の
「今後の拡張予定」を参照。
