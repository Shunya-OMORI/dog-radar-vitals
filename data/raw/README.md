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
  同意書署名は不要）。**被験者01-30全員を配置済み**（`datasets_subject_01_to_10/11_to_20/21_to_30_scidata.zip`
  の3ファイル、計約5.6GB）。2026-07-23、101・102モデルの5-fold cross-validationのため
  01-10から30名全員に拡張した。

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

## MMECGデータセット（mmWave radar、拍単位ECG同期、2026-07-24追加）

イヌのRR Interval・ECG波形予測という本題に、拍単位の正解付きデータセットで初めて直接取り組めるようになった。
Chen, Jinbo, et al. (2022)、*mmECG: Contactless Electrocardiogram Monitoring With Millimeter Wave Radar*
（TI AWR1843 mmWave radar + DCA1000、3Tx4Rxの仮想12chアレイ、フレームレート200Hz）。
ユーザから同意書契約済みの配布物 `MMECG202211.rar` を直接受け取り配置した（申請不要のSchellenbergerとは異なり、
機関印付き契約書の送付が必要な制限付き配布のデータセットである点に注意。再配布不可）。

- 参照実装（未取得、データ構造の説明のみREADMEで確認）: [github.com/jinbochen0823/RCG2ECG](https://github.com/jinbochen0823/RCG2ECG)
- 展開には `unrar`（RAR5形式）が必要。本環境にはapt経由でインストールできなかった（sudoパスワード必須）ため、
  RARLAB公式サイトの静的バイナリ（`https://www.rarlab.com/rar/rarlinux-x64-621.tar.gz`、インストール不要）を使って展開した。

```
data/raw/mmecg/
└── finalPartialPublicData20221108/
    ├── 1.mat ... 91.mat   # 91トライアル（.matファイル）
```

各`.mat`はMATLAB構造体`data`1つを含み、scipy.io.loadmat(struct_as_record=False, squeeze_me=True)で読める。フィールド:

- `RCG`: shape `(35505, 50)` float64 — 50点の3D心臓表面変位計測（Radar CardioGram）。fs=200Hz（35505サンプル≈177.5秒≈3分）
- `ECG`: shape `(35505,)` float64 — 同期ECG波形（真値、1リード）
- `posXYZ`: shape `(50, 3)` float64 — RCGの50点それぞれの3D座標
- `id`: int — 被験者ID、`gender`: str（'boy'/'girl'）、`age`: int
- `physistatus`: str — 生理状態。'NB'(通常呼吸)・'IB'(不規則呼吸)・'PE'(運動後)・'SP'(睡眠)

**注意: 91トライアルは11被験者分（id: 1, 2, 5, 9, 10, 13, 14, 16, 17, 29, 30）に集約される。**
1被験者あたり2〜23トライアルと非常に偏っている（id=29が23トライアルで最多）。
**トライアル単位ではなく被験者ID単位でtrain/val/test分割しないと個体リークが起きる**
（イヌデータのCVで実証済みの教訓と同型の罠。`AGENTS.md`参照）。11被験者という少なさは、
イヌ10頭のケースと同様、単一分割の結論を鵜呑みにできない規模であることを意味する。

現行の実装（`configs/experiments/201_*.yaml`〜、`EXPERIMENTS.md`参照）は、11被験者を
`train: [1,2,5,9,10,13,14]` / `val: [16]` / `test: [17,29,30]` に分割した単一splitから開始し、
後段でLeave-Subjects-Out cross-validationに拡張する方針（101/102と同じ育て方）。
