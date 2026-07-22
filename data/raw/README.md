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

## ヒトデータでの手法検証用データ（2026-07-22追加）

イヌのHR/BR予測モデルが「個体内の時間変動を全く追えていない」ことが判明したため
（[`../../EXPERIMENTS.md`](../../EXPERIMENTS.md) 参照）、レーダから他センサの**波形**（ECG等）を
推定する手法をヒトデータで先に検証する目的で追加した。

Schellenberger et al., *Scientific Data* 7:291 (2020)。健常者30名、24GHz CW radarと
臨床用ECG・インピーダンス心図・連続血圧を同期記録。

- 取得元: Figshare, DOI [10.1038/s41597-020-00629-5](https://doi.org/10.1038/s41597-020-00629-5)
  （データ本体は [Figshare 12186516](https://doi.org/10.6084/m9.figshare.12186516)）
- **申請不要でFigshare APIから直接ダウンロード可能**（イヌデータと異なり、MMECGのような
  同意書署名は不要）。現在は被験者01-10のみ配置（`datasets_subject_01_to_10_scidata.zip`）。
  被験者11-30が必要になれば同様に取得する。

```
data/raw/schellenberger_human/
├── GDN0001/
│   ├── GDN0001_1_Resting.mat    # scipy.io.loadmatで読める。フィールド:
│   ├── GDN0001_2_Valsalva.mat   #   radar_i, radar_q: レーダI/Q（fs_radar=2000Hz）
│   ├── GDN0001_3_TiltUp.mat     #   tfm_ecg1, tfm_ecg2: 同期ECG（fs_ecg=2000Hz、レーダと同一fs・同一長）
│   └── GDN0001_4_TiltDown.mat   #   tfm_bp, tfm_icg, tfm_z0: 血圧・インピーダンス心図等（fsは別）
├── GDN0002/ ... GDN0010/         # 同構造（GDN0004以降はApneaシナリオも追加である5ファイル）
```

**注意**: GDN0003の`Resting`ECGに41サンプル(20ms)のNaN欠損区間がある（センサ瞬断と推測）。
`data/ecg_windowing.py`のNaN検出で該当窓を自動的にスキップする。

現行の実装（`configs/experiments/101_ecg_cnn1d_resting.yaml`）はRestingシナリオのみを対象とする。
