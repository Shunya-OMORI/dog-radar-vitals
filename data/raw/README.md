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

## RawData_No*.csv の列構成（467列の内訳）

**原論文（Ahmed et al. 2024）本文には、CSVの列構成についての明示的な記述が無い。**
Data Recordsセクションは「レーダデータは9000サンプルから成る（The radar data file consists
of 9000 samples）」と行数（フレーム数）のみを述べ、1フレームあたりの列数や、列が何を表すかには
一切触れていない（PMC全文で "467"・"range bin"・"fast-time"・"observation window" に該当する
記述は見つからなかった）。以下は、原論文Methods節に明記された機材仕様（Table 1）と、実際に
配布されているCSVファイルの検証から導いた解釈であり、**論文の直接引用ではなく、こちらでの
技術的な裏付けに基づく推定**である。

- レーダは Novelda Xethru-X4（IR-UWB）、**サンプリング周波数 23.32 GHz**、**フレームレート 50 FPS**、
  送受信1対（Methods, Table 1）。IR-UWBレーダは、1回のパルス送信ごとに「高速時間（fast-time）」
  方向へ受信波形をサンプリングして1フレームを作り、それをフレームレートで繰り返し取得する
  （＝「低速時間（slow-time）」方向が時間経過）という構成を取る。
- **行 = フレーム（slow-time、50Hzで180秒分=9000行）**、**列 = 1フレーム内の高速時間サンプル
  （＝レンジビン、467列）** という対応が、上記の一般的なIR-UWBレーダの動作と、
  9000行という論文記載の値（=50FPS×180秒）に矛盾しない。
- 467列という値は、サンプリング周波数23.32GHzと組み合わせると次のように解釈できる:
  `467 / 23.32e9 ≈ 20.0 ns`。これを片道距離に換算すると `c × 20.0ns / 2 ≈ 3.0 m`
  （c=光速）となり、被験体からの距離0.3m（Methods）を十分内包する「観測窓（レンジウィンドウ）」
  の幅として自然な値になる。この一致は、467列が高速時間方向のレンジビンであることを
  裏付ける傍証と考えている（ただし論文にこの計算の明記は無く、こちらでの推定である）。
- **各列は単一の実数値**（正負混在、複素数のI/Q成分としてペアになった列は無い）。Novelda X4
  チップはベースバンドI/Qで出力するモードと、ダウンコンバートしない実数RFサンプルを直接
  出力するモードの両方をサポートするため、本データは後者（実数RFサンプル）に相当すると
  考えられるが、これも論文に明記はなく推定に留まる。
- **収録されている物理量はレーダ1系統のみ**であり、467列の中に複数の異なるセンサの生データが
  多重化されているわけではない。心電センサ(BM7Vet Pro)の生データは、レーダCSVには一切含まれず、
  別ファイルの`BIONET/Reference_HR_No*.csv`・`Reference_BR_No*.csv`（いずれも心電センサ内部で
  1FPSへ平均化・演算済みの心拍数・呼吸数の値であり、心電波形そのものではない）としてのみ存在する。

**要約すると**: `RawData_No1.csv`の467列は「複数センサの列」ではなく、**単一のUWBレーダが
1フレームごとに取得する高速時間方向の反射波形サンプル（レンジビン）**である。原論文はこの
列構成を明示していないため、上記は既知の機材仕様と配布データの整合性から導いた解釈である
ことに留意されたい。

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
