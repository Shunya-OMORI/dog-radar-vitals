# data/raw/

このディレクトリは `.gitignore` により中身が追跡されない（大容量かつ再取得可能なため）。

## 現在配置しているデータ

Ahmed et al., *Scientific Data* 11:107 (2024) の公開データセット「UWB-DVS」のシナリオ1（麻酔下のイヌ10頭）。

- 取得元: Figshare, DOI [10.6084/m9.figshare.23820915](https://doi.org/10.6084/m9.figshare.23820915)
- 詳細な書誌・実験条件は [manager-agent/research/dog-mmwave-rri/summaries/03_Ahmed-2024_dog-uwb-public-dataset.md](../../../manager-agent/research/dog-mmwave-rri/summaries/03_Ahmed-2024_dog-uwb-public-dataset.md) を参照。

```
data/raw/
├── Load.m                          # 原著者配布のMATLAB読み込みスクリプト（参考用、本リポジトリのPythonパイプラインでは未使用）
└── Scenario1_FaintedDogs/
    ├── No1/
    │   ├── Radar/RawData_No1.csv       # 9000行(50FPS×180秒) × 467列(レンジビン)
    │   └── BIONET/
    │       ├── Reference_HR_No1.csv    # 180行、1FPSの心拍数[bpm]
    │       └── Reference_BR_No1.csv    # 180行、1FPSの呼吸数[bpm]
    ├── No2/ ... No10/                  # 同構造
```

## 注意: このデータに含まれない情報

参照センサ（BM7Vet Pro、臨床承認済み獣医用ECGセンサ）の出力は **1FPSへ平均化された心拍数・呼吸数のスカラ値のみ**であり、
拍単位のRR IntervalもECG波形も含まれない（原論文Methods節、および上記要約の（p）節を参照）。
そのため本リポジトリの現行実装は **心拍数予測・呼吸数予測の2タスクのみ**を対象とする。
RR Interval・ECG波形予測は、拍単位の正解が取得できる別データセットが用意された時点で追加タスクとして拡張する
（[../../EXPERIMENTS.md](../../EXPERIMENTS.md) の「今後の拡張」参照）。

## 再取得手順

1. Figshareから `Scenario1_FaintedDogs.zip` をダウンロード
2. このディレクトリ (`data/raw/`) に展開し、上記の構造になるようにする
