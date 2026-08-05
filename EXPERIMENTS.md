# 実験ログ

このファイルは人間が手で維持する台帳である。`runs/` はgitignore対象なので、
「どのrunが何のためで、結果がどうだったか」はここに書かない限り失われる。
コーディングエージェントによる試行錯誤が増えても、ここを見れば経緯を追えることを目的とする。

**運用ルール**: 学習を1回走らせたら、成否によらず1行追加する。うまくいかなかった実験も
「何を試して何が起きなかったか」を残すことに価値がある。

## ログ

| 日付 | run_dir | 設定 | 変更点・狙い | 結果 | 次の一手 |
|---|---|---|---|---|---|
| 2026-07-22 | `runs/20260722-*_20260722_baseline`（12件、`reports/20260722_baseline/`に集約） | `configs/experiments/001`〜`012` 全件 | 深層3種（Transformer/CNN1D/LSTM、100 epoch、GPU）×古典ML3種（Ridge/RandomForest/GradientBoosting、`data/features.py`の手作り特徴量、CPU並列）を HR/BR 両タスクで一括比較。初回の本番実行 | **BR**: 全モデルがtest MAE 2.0〜2.2 bpmに収束（deep系がわずかに優位、transformer 2.000が最良）。**HR**: 古典ML3種が18.7〜18.9 bpmで最良、CNN1D/LSTMが19.6程度、Transformerが31.0で最下位。詳細は[`reports/20260722_baseline/table.md`](reports/20260722_baseline/table.md)・[`chart.png`](reports/20260722_baseline/chart.png) | 下の「本実行から分かったこと」参照 |
| 2026-07-22 | `runs/20260722-1002xx〜1021xx_hr_transformer_20260722_hr_transformer_ablation`（4件、`reports/20260722_hr_transformer_ablation/`に集約） | `configs/experiments/013`〜`016` | 001のTransformer(HR)が100 epochで未収束だった件を受け、学習率・エポック数・スケジューラの対照実験。013: lr 1e-4→5e-4。014: lr 1e-4→1e-3。015: lrは1e-4のままepoch 100→300。016: 10epochウォームアップ+コサイン減衰（ピークlr 5e-4）、200epoch | 4件とも001(test MAE 31.0)から大幅改善: 013=17.45（最良、古典MLの18.7を上回る）、014=19.48、015=18.74、016=17.83。学習曲線（[`learning_curves.png`](reports/20260722_hr_transformer_ablation/learning_curves.png)）で001の学習率が単に低すぎたことを確認。ただし**別の問題が判明**（下記参照） | 下の「HR Transformer対照実験の結果」参照 |
| 2026-07-22 | `runs/20260722-153929_ecg_ecg_cnn1d`（`reports/20260722_ecg_resting/`に集約） | `configs/experiments/101_ecg_cnn1d_resting.yaml` | イヌHR/BRでの個体内相関ほぼゼロという結果を受け、目標を「スカラ値予測」から「他センサ波形推定」に転換する手法検証。Schellenberger et al. (2020) ヒトデータセット（24GHz CW radar、fs=2000Hzで同期したECG）でレーダI/Q→ECG波形をsequence-to-sequenceで推定する全畳み込み1D CNNを実装、Restingシナリオ7/1/2名で学習 | test_corr=0.524（test被験者2名個別でも0.459・0.580と一貫）。train_corr=0.905まで到達し、val_corr(0.47〜0.51)との間に過学習傾向あり。波形を可視化すると心拍のタイミング（R波の出現位置）はおおむね捉えているが、QRSの鋭い振幅・形状の再現は弱い（[`waveform_example.png`](reports/20260722_ecg_resting/waveform_example.png)） | 下の「ヒトECG波形推定（手法検証）の結果」参照 |
| 2026-07-23 | `runs/_cross_validation/cv10_013_transformer_hr.json`, `cv10_009_random_forest_hr.json` | 013・009をLeave-One-Dog-Out CV（n_folds=10）で再検証、`scripts/test_cv_significance.py`で対応のある検定を追加 | 013: 3/10foldでtrivial超え、平均差+5.13（悪化）、Wilcoxon片側p=0.92。009: 3/10fold、平均差+0.53、p=0.84。**5foldの結果と一致し、trivialとの差は統計的に有意ではないことが10foldでも裏付けられた** | 下の「Leave-One-Dog-Out CVと統計検定」参照 |
| 2026-07-23 | `runs/20260723-003215_ecg_rpeak_cnn1d`（`reports/20260723_ecg_vs_rpeak/`に集約） | `configs/experiments/102_rpeak_cnn1d_resting.yaml` | RR Intervalは波形再構成と定式化・アーキテクチャが異なるはず、というユーザ仮説を検証。101と同一データ・split・windowで、密な波形回帰ではなく「R波位置のガウシアンheatmap」を回帰する別モデル(sigmoid出力+BCE損失)を試作 | test_corr=0.335(heatmap相関)。101(ecg_cnn1d)との比較は下記「101 vs 102」参照。R波検出F1・RR Interval誤差でみると、101は被験者間で大きく変動(F1 0.08〜0.52)する一方、102はより安定(F1 0.32・0.32、RR MAE 9.6ms・12.2ms) | 下の「101 vs 102: 波形回帰とheatmap回帰の比較」参照 |
| 2026-07-23 | `runs/_cross_validation/ecg_cv_20260723.json`（`reports/20260723_ecg_vs_rpeak/cv_paired_scatter.png`に集約） | 101・102を5-fold CVで再検証。被験者11-30を追加取得し全**30被験者**に拡大 | F1(n=30): ecg_cnn1d 0.504±0.171 vs rpeak_cnn1d 0.474±0.237（Wilcoxon p=0.289、有意差なし）。RR Interval MAE(n=30): 16.0ms vs 14.0ms（**Wilcoxon p=0.033、有意**）。訓練被験者を7→23名に増やしたことでF1自体も大きく改善した | 下の「101 vs 102のcross-validation（全30被験者）」参照 |
| 2026-07-25 | `runs/20260725-005323_rr_interval_ridge`, `runs/20260725-005342_rr_interval_random_forest`, `runs/20260725-005403_rr_interval_gradient_boosting` | `configs/experiments/211`〜`213`（MMECG、古典ML3種、RR Interval[ms]直接回帰） | ユーザから拍単位正解付きmmWaveデータセットMMECG(Chen et al. 2022)を入手し、201番台としてRR Interval・ECG波形予測の本実装を開始（詳細は下記「MMECGでのRR Interval・ECG波形予測: 実装フェーズ」参照）。まず古典ML3種（Ridge/RandomForest/GradientBoosting、心拍帯域パワー最大チャネルの手作り特徴量）で単一split（train7/val1/test3被験者）を実行 | val_mae 47.5/49.0/50.5ms、test_mae 94.5/95.4/96.9ms。val→testでの大幅な悪化（約2倍）は、11被験者という少なさによる汎化ギャップの兆候であり、犬データで見た「単一分割の結論は再現しない」という教訓と整合する | 深層モデル群（201-210, 214-215）の学習完了後、`scripts/compare_mmecg_models.py`で横断比較しRR Interval MAEを揃えて評価する |
| 2026-07-25 | `runs/20260725-01*_ecg_*`, `runs/20260725-02*_ecg_*`, `runs/20260725-02*_beatgraph_*`（12件、`reports/mmecg_comparison/comparison_full.json`に集約） | `configs/experiments/201`〜`210`, `214`〜`215`（MMECG、CNN/Transformer/LSTM/NCP/複素CNN×波形回帰・heatmap回帰、GNN/GAN×ビートグラフ回帰） | 3つの本命（NCP・複素領域モデル・GNN(+GAN)）を通常DNNベースライン（CNN/Transformer/LSTM）・古典ML(211-213)と同一split・同一window設定で一括比較。NCPのみ学習コスト（1epoch≈5.5分）のためepochs 50→15 | RR Interval MAE最良はrpeak_cnn1d(10.7ms、本命ではないbaseline)。ecg_ncp(21.1ms)はecg_cnn1d(21.2ms)とほぼ同着、rpeak_ncp(30.0ms、学習中に出力が定数へ退化)は深層最悪。複素CNN・GNN/GANもいずれも対応するbaselineへの明確な優位を示さず。rpeak_transformerは未収束(F1=0) | 下記「MMECGでのRR Interval・ECG波形予測: 実装フェーズ」の「総括」参照 |
| 2026-07-28 | `runs/20260728-013817_ecg_ecg_cnn1d` | `configs/experiments/240_mmecg_cnn1d_minmax.yaml`（201のnormalizationをzscore→minmaxに変更のみ） | Chen et al.原著（相関中央値0.90）の再現に向けた第一歩。ユーザ経験則（IMU/レーダ入力・ECG出力ともmin-max[-1,1]正規化が良い）と`docs/Radar2ECG_*.pdf`の前処理方針を踏まえ、`data/mmecg_windowing.py`にminmax正規化を追加し201と対照比較。あわせて`scripts/evaluate_per_beat_correlation.py`を新設し、Chen et al.の評価方法（拍単位でセグメント化してから相関、中央値報告）でも再評価した | 窓内相関はminmax 0.212 vs zscore 0.224でほぼ差なし。拍単位相関（Chen式、中央値、n≈19,227拍）はminmax 0.216 vs zscore 0.163で、依然0.90には遠く及ばない。**正規化方式の変更も、評価粒度をChen et al.に揃えることも、単独ではギャップを埋めなかった** | 下記「相関0.90再現の第一歩」参照。次はTCN自己回帰デコーダ（過去ECGサンプルをデコーダ入力に含める）とμ-law companding+カテゴリカル損失の実装を優先する |
| 2026-07-28 | `runs/20260728-020912_ecg_chen2022_reconstructor` | `configs/experiments/241_mmecg_chen2022_repro.yaml`（新規`models/deep/chen2022_reconstructor.py`, `training/chen2022_trainer.py`, `data/mulaw.py`） | ユーザから原著の完全再現を明示的に要望され、残っていた2要素（TCN自己回帰デコーダ、μ-law companding+256値カテゴリカル損失）を実装。既存の`ecg_spatial_fusion`(posXYZ使用)を空間時間エンコーダとして転用し、dilation 2^i(i=0..8)のcausal TCNデコーダで自己回帰生成。訓練は教師強制で並列、評価は原著と同じ自己回帰生成 | 教師強制でのcross-entropy lossは順調に収束(train 3.92→0.89, val→1.50)したが、**自己回帰生成での再評価は窓内相関0.065・拍単位相関中央値0.014と、既存のfeedforward回帰(201: 0.224/0.163)より明確に悪化**。診断の結果、教師強制下の高精度(相関0.996)は「RCGを無視して1つ前の正解サンプルをそのままコピーするだけ」のtrivial baselineが既に相関0.977に達することで説明できてしまい、モデルがRCGから学習した証拠にはなっていなかった。自己回帰生成に切り替えるとこのショートカットが使えず、誤差が蓄積して波形が長時間停滞する現象を確認した | 下記「Chen et al.アーキテクチャの忠実な再現を実装・実行した結果」参照。**原著の0.90は現状のデータ規模(訓練7人)では再現不可能という結論をさらに強化。** 自己回帰系モデルの評価では今後必ずtrivial copy-prev-sample baselineを併記する |
| 2026-07-28 | - (調査のみ、学習runなし) | `data/raw/MMECG202211.rar`の`unrar lb`による全件確認、GitHub `jinbochen0823/RCG2ECG`のREADME精読、radarODE(arXiv:2408.01672)本文精読 | ユーザから「rarの中身は本当に原著より少ないのか」「原著を超える論文があれば所属研究室とデータセットを調べよ」との指摘を受け、直前のセッションの結論（データ量がボトルネック）を再検証 | **結論を訂正: 11被験者は取得漏れではなく原著者配布の全量(4.55h、GitHubに明記)であり、かつ「11被験者では0.90は再現不可能」という前回の結論は誤りだった。** 無関係の研究室(XJTLU/HKUST-GZ)によるradarODEが**我々と全く同じ91トライアル・11被験者**を使い、11-fold leave-one-subject-out CVで、Chen et al.の再現アーキテクチャでPCC87.9%、radarODE自身で92.6%を達成していることを確認（Table VI）。ボトルネックはデータ量ではなく、生時系列のまま入力する信号処理・複数拍窓を一括回帰するタスク設計・振幅を潰す損失関数・自己回帰生成の罠、という方法論側にあった | 下記「データ規模の再検証」参照。次はSST(時間周波数変換)による信号処理の高度化、1心拍単位へのタスク再定式化、ODE型振幅事前分布の3点を実装する |
| 2026-07-28 | `runs/20260728-044928_ecg_radarode_sceg`, `runs/20260728-055930_ecg_radarode_longterm` | `configs/experiments/242_mmecg_radarode_sceg.yaml`, `243_mmecg_radarode_longterm.yaml`（新規`data/sst.py`, `data/mmecg_sst_cache.py`, `data/mmecg_ppi.py`, `data/mmecg_singlecycle_dataset.py`, `data/mmecg_radarode_longterm_dataset.py`, `models/deep/radarode_sceg.py`, `models/deep/radarode_longterm.py`） | ユーザ指示によりradarODEの再現を実装。SST変換(91トライアル事前計算・`data/processed/mmecg_sst/`にキャッシュ)、50ch多数決PPI推定、SCEG(単一拍ECG生成、McSharry型ODEデコーダ内蔵)、長期再構成(非自己回帰9層TCN融合)の2段パイプライン | **目標(PCC 87.9〜92.6%)に届かず**: SCEG単一拍相関0.147、長期再構成窓内相関0.163(201の0.224にも届かず)。学習曲線は強い過学習(train corr 0.98超、val corrは0.15〜0.22で頭打ち)。原因はレシピの誤りではなく、システムRAM(15GB)制約でSSTセグメントの時間分解能を`T_FIXED_SST=16`まで削らざるを得なかったこと(一度メモリ枯渇寸前で学習を強制終了した経緯あり)、Deformable Conv2d不在(torchvision未導入)、PPI推定の簡略化、が主因と考えられる | 下記「radarODEの実装・再現結果」参照。次はユーザ指示によりECG基盤モデル/事前学習モデルを使った形状事前分布の調査に移る |

## 本実行から分かったこと（2026-07-22）

- **BRタスクはモデル間でほぼ差がつかない。** 全6モデルがtest MAE 2.0〜2.2 bpmの狭い範囲に
  収まった。原因として、犬ごとのBR参照値がほぼ一定（`data/raw/README.md`記載の通り、
  1頭内では15.5や18.5のような単一値に近い）ため、この課題は実質「テスト犬の個体差を
  当てる」問題に近く、モデルの表現力による差が出にくいと考えられる。BRタスクでの
  モデル比較は現状あまり情報を持たない可能性がある。
- **HRタスクではTransformerが明確に劣後した（31.0 vs 他モデル18.7〜19.6）。**
  学習曲線を確認すると、train MAEは100 epoch終盤でほぼ0（0.01〜0.02）まで低下した一方、
  val MAEは31付近から単調減少を続けたまま100 epochで打ち切られており、**収束前に学習を
  止めてしまっている**（過学習ではなく学習不足）。CNN1D・LSTMは同じepoch数・同程度の
  学習率で19.6まで収束しており、Transformerだけがより多くのepochを要する可能性が高い。
- **古典MLモデル（`data/features.py`の9次元手作り特徴量、Ridge/RandomForest/GradientBoosting）が
  HRタスクで最良だった。** 深層モデルが生の467次元レンジビン系列を直接処理するのに対し、
  古典MLは分散最大のレンジビンを選んで1次元信号に落とし、時間統計量とバンドパワーに
  要約した特徴量を使っている。サンプル数が少ない（train 7頭分、約1200窓）ため、
  次元を絞った特徴量表現の方が有利に働いた可能性がある。

### 次に試すこと

- ~~Transformer(HR)のepoch数を増やす（200〜300程度）か、学習率スケジューラ（warmup+decay）を
  導入し、CNN1D・LSTMと同等以上の水準まで収束させてから比較をやり直す。~~
  → 2026-07-22実施済み。下記「HR Transformer対照実験の結果」参照。
- BRタスクは、犬ごとの参照値がほぼ定数であるという性質を踏まえ、「テスト犬の識別＋
  オフセット予測」のような定式化に切り替えるか、より変動の大きい別データセットで
  再評価することを検討する。
- 古典MLがHRで優位だった要因が「特徴量設計」なのか「サンプル数に対するモデル容量」
  なのかを切り分けるため、深層モデルにも同じ特徴量（`data/features.py`の出力）を
  入力するアブレーションを追加してもよい。
- ~~【最優先】train/val/testの犬IDによる単一分割（7/1/2頭）の信頼性を疑う。~~
  → 2026-07-22実施済み。下記「犬入れ替えcross-validationの結果」参照。
  **疑いは的中し、単一分割での「モデルAが優れている」という結論はいずれも再現しなかった。**
- ~~HRでtrivialを上回った013・016は健康モニタリングに使えるか~~ → 2026-07-22検証済み。
  「使えない」が結論（下記「追試: HRでtrivialを上回った2モデルは健康モニタリングに使えるか」参照）。
  個体内の時間変動を全く追えておらず、母集団平均への回帰にすぎない。
- **現行のScenario1（麻酔下・3分間・1頭1回）というデータ構造そのものが、健康モニタリングの
  検証に不向きである。** 個体の状態変化を検知するには同一個体の複数時点データが要るが、
  現行データにはそれが存在しない。Scenario2（覚醒・自由行動・30分、ただし臨床参照なし）や、
  別データセット・自前データ取得の検討が必要。

## 犬入れ替えcross-validationの結果（2026-07-22）

ユーザから「犬を入れ替えた学習（1匹の犬に過学習しない仕組み）も試してほしい」との依頼を受け、
`scripts/run_dog_cross_validation.py` を実装した。10頭をシャッフルして5foldに分割
（各foldでtest犬2頭、残り8頭のうち1頭をval・7頭をtrain）し、foldごとに
`compute_baselines()`でtrivial_maeを算出し直して比較する
（fold構成によって難易度＝trivial_maeの水準そのものが変わるため、fold横断でtrivial比を
見ることが重要）。

対象は、単一分割での「勝者」だった2モデル: **013 transformer(lr=5e-4)**（深層側の最良）と
**009 random_forest**（古典ML側の最良）。いずれもHRタスク。

| model | mean test_mae | std | trivialを上回ったfold数 |
|---|---|---|---|
| transformer (013) | 14.591 | 6.947 | 2/5 |
| random_forest (009) | 11.270 | 2.517 | 1/5 |

（[`reports/20260722_dog_cross_validation/cv_vs_trivial.png`](reports/20260722_dog_cross_validation/cv_vs_trivial.png)
にfold毎のmodel MAE vs trivial_maeの棒グラフあり）

### 結論

**単一分割で見えていた「勝者」は、犬を入れ替えると再現しなかった。**

- **013 (transformer, lr=5e-4)**: 単一分割ではtest_mae=17.45でtrivial(18.8)を+7.3%上回り
  16モデル中最良だった。しかしCVでは5fold中3foldでtrivialを下回り（fold1: 23.47 vs
  trivial 13.57、fold2: 15.91 vs 12.92、fold4: 20.36 vs **trivial 5.17**＝trivialの4倍近い
  誤差）、fold間のばらつき（std=6.95）も非常に大きい。単一分割での勝利は、たまたま
  「trivialを上回りやすいtest犬の組」（No9, No10）に当たっただけだった可能性が高い。
- **009 (random_forest)**: 単一分割ではtest_mae=18.74でtrivial(18.8)とほぼ同点だった。
  CVでも同様の傾向で、5fold中4foldでtrivialと同等かそれ以下（fold1: 14.98 vs 13.57、
  fold2: 13.33 vs 12.92、fold3: 9.91 vs 9.44、fold4: 8.06 vs **trivial 5.17**）であり、
  「trivialベースラインを安定して上回る」という主張はできない。std=2.52とtransformerより
  変動は小さいが、これは「常にtrivial付近に留まる」ことの裏返しでもある。
- **どちらのモデルも、5fold合計10回の（モデル, fold）組のうち、trivialを明確に
  上回ったのは3回のみ**（013が2回、009が1回）。残り7回は同等かそれ以下だった。

### 含意

本ファイルのこれまでの記述にある「013・016がHRでtrivialを上回った」「古典MLが優位」
「lr=5e-4が最良のTransformer設定」といった結論は、**いずれも単一のtrain/val/test分割
（val=No8, test=[No9,No10]）に固有の結果であり、犬を入れ替えると一般には再現しない。**
今後のモデル比較・ハイパーパラメータ探索は、単一分割ではなく本節のcross-validation手順
（`scripts/run_dog_cross_validation.py`）を標準とし、fold平均とfold毎のtrivial比較を
セットで報告する運用に切り替える。単一分割の結果（001〜016の各エントリ）は、
初期実装の動作確認・大まかな見通しとしての価値はあるが、「どのモデルが優れているか」の
根拠としては扱わない。

### 次に試すこと

- 他の設定（016 warmup、CNN1D、LSTM、Ridge、Gradient Boosting等）もCVで検証し、
  10モデル全体でfold平均・fold毎trivial比較を揃える。
- foldごとのtrivial_maeの水準差（5.17〜13.57）自体が、犬の個体差（体格・犬種による
  安静時心拍数のばらつき）の大きさを表しており、この母集団内変動を縮小するには
  対象個体数を増やすか、体重・犬種等の付帯情報を特徴量に含める設計を検討する余地がある。
- n_folds=5・test 2頭/foldは10頭という個体数からくる制約であり、統計的検出力は本質的に
  低い。イヌの拍単位データを自前で追加取得する場合、個体数の確保を精度以上に優先すべき
  である。

## Leave-One-Dog-Out CVと統計検定（2026-07-23）

ユーザから「連続量には明確なチャンスレベルが無いので、誤差が大きいか小さいか判断しづらい」
との指摘を受け、2点を追加した。

1. **n_folds=5(test 2頭/fold)→n_folds=10(test 1頭/fold, Leave-One-Dog-Out)へ変更。**
   `run_dog_cross_validation.py`は`--n-folds 10`を渡すだけで対応（コード変更不要、10頭で
   n_folds=10なら自動的に1fold=1頭になる）。各foldの訓練犬が7頭→8頭に増え、評価点も
   5個→10個に増える。
2. **`scripts/test_cv_significance.py`を新設し、fold毎の(model_mae, trivial_mae)の
   対応のある差を符号検定・Wilcoxon符号順位検定で検定する。** 連続量の絶対誤差だけでは
   「良い/悪い」の基準がないため、trivialとの対応のある比較を検定に落とし込むことで
   「たまたま良く見えているだけ」なのかを判定できるようにした。

### 結果

| model | n_folds | trivialを上回ったfold数 | 平均差(model-trivial) | 符号検定p値 | Wilcoxon p値(片側) |
|---|---|---|---|---|---|
| transformer (013, lr=5e-4) | 5 | 2/5 | +4.097 | 1.000 | 0.844 |
| transformer (013, lr=5e-4) | **10 (LODO)** | 3/10 | **+5.125** | 0.344 | 0.920 |
| random_forest (009) | 5 | 1/5 | +0.777 | 0.375 | 0.906 |
| random_forest (009) | **10 (LODO)** | 3/10 | **+0.530** | 0.344 | 0.839 |

**foldをtest 2頭→1頭に増やし、対応のある検定を追加しても結論は変わらなかった。**
両モデルとも、trivialとの平均差はプラス（trivialより悪い）であり、符号検定・Wilcoxon
検定のいずれもp値は0.3以上（有意水準0.05を大きく超える）。10頭のうち、trivialを
明確に上回ったのは各モデルとも3頭のみで、これは偶然の範囲を出ない。

**結論: 現行データ（イヌ10頭、麻酔下、Scenario1）でのHR予測は、モデルをどう選んでも
trivialベースライン（訓練犬の平均値を常に返すだけ）を統計的に有意には上回れていない。**
この結果は、モデル実装の問題というより、(a) oracle_mae≈1bpmが示す通り個体内変動が
ほぼ無く学習すべき動的信号が乏しいこと、(b) train8頭・test1頭という個体数の少なさが
統計的検出力を本質的に制限していること、の2点に起因すると考えられる
（[`reports/progress_report_2026-07-22.md`](reports/progress_report_2026-07-22.md)の
解釈も参照）。モデル改良より個体数確保を優先すべきという結論を、より強い根拠とともに
再確認した。

## 101 vs 102: 波形回帰とheatmap回帰の比較（2026-07-23）

ユーザから「RR Interval予測は一般的に問題設定もアーキテクチャも異なるはず」との指摘を受け、
101(ecg_cnn1d、密な波形振幅を回帰)とは別に、**R波の位置だけを疎なイベントとして回帰する
モデル**を試作した（`configs/experiments/102_rpeak_cnn1d_resting.yaml`）。

- ターゲット: 真のECGからR波を検出し（`data/rpeaks.py`の閾値+不応期検出器）、
  各R波位置にガウシアン(σ=10ms)を立てて重ね合わせたheatmap（値域[0,1]）
- モデル: 101と同じ全畳み込みバックボーンだが、出力層にsigmoidを追加し、損失もMSEから
  BCEに変更（`models/deep/rpeak_cnn1d.py`）
- 評価: 101と102の予測（波形 or heatmap）それぞれから改めてR波を検出し、真のR波との
  タイミング一致度（±50ms許容、F1）とRR Interval誤差（マッチしたペアのみ、MAE[ms]）を
  共通の物差しで比較（`scripts/compare_ecg_vs_rpeak.py`、`data/rpeaks.py`の`match_peaks`）

### 結果

| 被験者 | モデル | precision | recall | F1 | RR Interval MAE |
|---|---|---|---|---|---|
| GDN0009 | 101 ecg_cnn1d（波形） | 0.089 | 0.075 | 0.081 | 7.2ms（マッチ数少なくノイジー） |
| GDN0009 | 102 rpeak_cnn1d（heatmap） | 0.581 | 0.223 | **0.323** | 9.6ms |
| GDN0010 | 101 ecg_cnn1d（波形） | 0.647 | 0.439 | **0.523** | 26.7ms |
| GDN0010 | 102 rpeak_cnn1d（heatmap） | 0.718 | 0.208 | 0.322 | 12.2ms |

（[`reports/20260723_ecg_vs_rpeak/comparison.png`](reports/20260723_ecg_vs_rpeak/comparison.png)）

### 解釈

**両アプローチともR波検出としては絶対水準は低い（F1 0.08〜0.52）が、性質が明確に異なる。**

- **101(波形回帰)は被験者間で結果が大きく振れる**（F1 0.081 vs 0.523、RR MAE 7.2ms vs
  26.7ms）。以前の可視化で確認した通り、QRSの鋭い振幅の再現度は被験者ごとにばらつきが
  大きく、それがそのままピーク検出の成否に直結していると考えられる。
- **102(heatmap回帰)は被験者間で相対的に安定している**（F1 0.323 vs 0.322、RR MAE
  9.6ms vs 12.2ms）。振幅そのものを再現する必要がなく「ここにR波があるはず」という
  確信度だけを学習するため、被験者ごとの波形の個人差に対して頑健になっていると考えられる。
  ただしrecallは両被験者とも0.2程度に留まり、検出漏れは多い。
- **RR Interval誤差（マッチしたペアのみ）は102の方が値の範囲が狭く安定している**
  （9.6〜12.2ms vs 7.2〜26.7ms）。101のGDN0009における7.2msは、マッチ数が少ない
  （recall 0.075、約48拍）ことによる少数サンプルの見かけ上の低さである可能性が高く、
  額面通りには受け取れない。

**「RR Interval予測は問題設定・アーキテクチャが異なる」というユーザの見立ては支持された。**
少なくともこのヒトデータでの試作では、イベント検出（heatmap回帰）に定式化し直すことで
被験者間の安定性が向上する傾向が見えた。ただし両者とも本番水準には遠く、この比較自体も
test被験者2名のみに基づく予備的な結果である点には注意が必要。

### 次に試すこと

- ~~test被験者を増やす（被験者11-30を追加取得）、または101・102双方をcross-validationで
  再評価し、この傾向が2名だけの偶然でないかを確認する。~~ → 2026-07-23実施済み。
  下記「101 vs 102のcross-validation（全30被験者）」参照。
- 102のrecallの低さ（0.2程度）を改善するため、heatmapのσ（現在10ms）や検出閾値、
  クラス不均衡に強い損失（focal loss等）を検討する。
- RR Interval誤差の評価を「マッチしたペアのみ」ではなく、見逃し・過検出も加味した
  総合指標（例: 一定時間窓内の平均心拍数のMAE）でも別途評価し、用途に応じた指標を選ぶ。

## 101 vs 102のcross-validation（全30被験者、2026-07-23）

上記の2被験者だけの比較が偶然でないかを確認するため、Schellenbergerデータセットの
被験者11-30を追加取得し（Figshareから申請不要で即時ダウンロード）、**全30被験者**で
`scripts/run_ecg_cross_validation.py`による5-fold cross-validationを実施した。

- fold構成: 被験者をシャッフルして5fold(各fold test 6名)に分割、残り24名のうち
  val 1名・train 23名。101・102とも同一のfold構成で学習（比較のペアが対応するように）。
- 評価ロジックは`compare_ecg_vs_rpeak.py`と共有化（`rpeak_evaluation.py`に切り出し）。
- 各(fold, test被験者)の組についてF1・RR Interval MAEを求め、30被験者ぶんの対応のある
  ペアに対してWilcoxon符号順位検定を実施。

### 結果

| 指標 | ecg_cnn1d（波形） | rpeak_cnn1d（heatmap） | Wilcoxon p値 |
|---|---|---|---|
| F1（n=30） | 0.504 ± 0.171 | 0.474 ± 0.237 | 0.289（有意差なし） |
| RR Interval MAE（n=30） | 16.0ms | 14.0ms | **0.033（有意）** |

（散布図: [`reports/20260723_ecg_vs_rpeak/cv_paired_scatter.png`](reports/20260723_ecg_vs_rpeak/cv_paired_scatter.png)。
対角線より下がheatmap回帰の勝ち）

### 解釈

**訓練データを7名→23名に増やしたことで、両モデルのF1が大きく改善した**
（前回の2被験者テストでは101が0.08〜0.52、102が0.32・0.32だったのに対し、
今回はいずれも平均0.47〜0.50まで底上げされた）。これは、このタスクにとって
訓練被験者数がボトルネックの一つだったことを示唆する。

**F1（検出できたか）では両モデルに有意差はない（p=0.289）が、RR Interval MAE
（検出できた拍の間隔がどれだけ正確か）ではheatmap回帰(102)が統計的に有意に優れている
（14.0ms vs 16.0ms, p=0.033）。** これは前回の予備的な観察（heatmap回帰の方がRR誤差が
安定して小さい）を、n=2からn=30に増やした上でより確からしい形で再確認したことになる。

**含意**: 「拍を検出できるかどうか」自体はどちらの定式化でも同程度だが、**検出した拍の
タイミング精度（RR Interval用途で本質的に重要な指標）は、疎なイベント検出として定式化する
方が優れる**という、ユーザの当初の見立てを支持する結果が、統計的に有意な形で得られた。
拍単位のタイミング推定を目的とするなら、密な波形再構成よりheatmap／イベント検出型の
アプローチを優先すべきという設計指針が得られた。

### 次に試すこと

- 30名でもF1は0.5前後に留まり、臨床応用に耐える水準ではない。訓練データをさらに増やす
  （Valsalva/TiltUp/TiltDown等、他シナリオも学習に含める）か、モデル容量・アーキテクチャ
  （U-Net的な多重解像度構造等）を見直す。
- RR Interval MAE(14〜16ms)がheatmap回帰で有意に改善した要因（振幅を無視できることの
  効果か、BCE損失の勾配特性か）を、アブレーションで切り分ける。
- この知見をイヌのデータに転用するには、イヌ側の拍単位正解データの取得が依然として前提となる
  （`data/raw/README.md`参照）。

## 【最重要】trivialベースラインとの比較（2026-07-22）

「MAEはどのくらいなら理想的か」を判断するため、`scripts/compute_baselines.py` で2つの
物差しを計算した（[`reports/baselines.md`](reports/baselines.md)）。

- **trivial_mae**: レーダを一切使わず、train犬の目的変数の平均値を常に予測した場合のMAE。
  モデルがこれを十分下回らなければ「レーダから何も学習していない」のと区別がつかない。
- **oracle_mae**: 各test犬の真の平均値を知っていたと仮定した場合のMAE。個体差を除いた
  窓内変動のみに起因する理論的な下限。

| task | train_mean | test_range | trivial_mae | oracle_mae |
|---|---|---|---|---|
| hr | 115.13 | [102.0, 143.0] | **18.819** | 0.986 |
| br | 17.21 | [15.5, 19.5] | **2.000** | 0.000 |

これまでの全16モデル（001〜016）をtrivial_maeと比較すると:

| task | trivial_maeを上回った(改善した)モデル | trivial_maeと同等以下(改善なし) |
|---|---|---|
| **hr** | 013 lr=5e-4 (17.45, +7.3%)、016 warmup (17.83, +5.3%) のみ、明確な改善 | 009 random_forest・015 ep300・007 ridge は trivial比+0.2〜0.4%でほぼ誤差範囲。011 gradient_boosting・014 lr=1e-3・003 cnn1d・005 lstm・001 baseline は **trivialより悪い** |
| **br** | **1件もなし** | 002 transformer(baseline)が2.000でtrivialと完全一致。残る5モデル(lstm/cnn1d/ridge/random_forest/gradient_boosting)は**全てtrivialより悪い** |

### 結論

**BRタスクは、これまで試した6モデルのうち1つも「レーダの単純な平均予測」を上回れていない。**
oracle_mae=0.000（犬ごとにBR値が完全な定数のため）という事実と合わせると、このタスクは
現状の定式化では実質的に学習すべき信号がほとんど無く、モデル比較として機能していない。

**HRタスクも、明確にtrivialを上回ったのは16モデル中2つ（013, 016、いずれもTransformerの
学習率を上げた変種）のみで、改善幅も+5〜7%にとどまる。** これまでの「古典MLがHRで優位」
という結論（本ファイル冒頭の「本実行から分かったこと」）は、実際には**古典MLがtrivialベースラインと
ほぼ同じ性能だっただけ**であり、「古典MLが優れている」のではなく「深層モデルの大半がtrivialにすら
届いていない」と読み替えるべきである。

### 追試: HRでtrivialを上回った2モデル(013, 016)は健康モニタリングに使えるか

ユーザから「trivialを明確に上回った013・016は健康モニタリングに有用か」という質問を受け、
`scripts/diagnose_within_dog_signal.py` を作って検証した。健康モニタリング（個体の状態変化を
検知する用途）に使うには、単に「母集団平均に近い値」を返すのではなく、**個体内の時間変動を
実際に追えている**必要がある。test犬2頭それぞれについて、真値の時間変動と予測値の時間変動の
相関（within_dog_corr）を計算した。

| run | dog | true_mean | true_std | pred_mean | pred_std | bias | within_dog_corr |
|---|---|---|---|---|---|---|---|
| 013 (lr=5e-4) | No9 | 103.71 | 0.935 | 117.16 | 3.016 | **+13.45** | 0.035 |
| 013 (lr=5e-4) | No10 | 141.35 | 1.472 | 119.89 | 3.132 | **-21.46** | 0.173 |
| 016 (warmup) | No9 | 103.71 | 0.935 | 115.84 | 1.769 | **+12.13** | -0.120 |
| 016 (warmup) | No10 | 141.35 | 1.472 | 117.82 | 1.595 | **-23.53** | 0.057 |

**結論: 有用ではない。** 4パターンとも within_dog_corr は -0.12〜0.17 とほぼゼロで、
個体内の時間変動を全く追えていない。さらに重要なのは予測値そのものの挙動である。
両モデルとも、HRが低いNo9（真値103.7）に対しては**+12〜13 bpm過大予測**、HRが高いNo10
（真値141.4）に対しては**-21〜24 bpm過小予測**という、母集団平均（train_mean=115.13）へ
強く引き寄せられる回帰（regression to the mean）を示した。つまりモデルは「レーダから
その犬固有の値を読み取っている」のではなく、**ほぼ一定の値（115〜120付近）を出力しているだけ**
であり、それがたまたまNo10側の誤差を大きく減らしたためtrivial全体平均を上回った、という
偶然の産物である可能性が高い。test犬が2頭しかないため、この「trivialを上回った」という
結果自体、統計的な裏付けに乏しい。

**含意**: 現時点のモデルは「未知の犬が来たときに、それらしい心拍数域を当てずっぽうより
少しマシに言い当てる」以上のことをしていない。健康モニタリングが本来必要とする
「その犬自身の平常時からの逸脱を検知する」という機能は、(a) 個体内の時間変動を追えていない、
(b) そもそも麻酔下3分間という単発スナップショットのデータには同一個体の複数時点データが
存在しないため、現状の評価設計では原理的に検証しようがない。

## ヒトECG波形推定（手法検証）の結果（2026-07-22）

ユーザからの提案で、目標をHR/BRという**スカラ値の予測**から、**他センサの時系列波形の推定**へ
広げる方向を検討した。イヌのデータには加速度・ECG波形の正解が一切存在しない
（`data/raw/README.md`参照）ため、まずヒトの公開データセットで手法を検証することにした。

MMECG（Chen et al. 2022、ミリ波、ヒト）は同意書署名＋メール申請が必要で承認まで約1週間かかり
即時には使えなかった。代わりに、申請不要でFigshareから即時ダウンロードできる
**Schellenberger et al. (2020)** データセット（24GHz CW radar、健常者30名、ECG・
インピーダンス心図・連続血圧を同期記録、レーダI/QとECGが**同一サンプリングレート2000Hz・
同一長で記録**されている）を使うことにした。

### 実装

- `data/schellenberger.py`: `.mat`ファイルの読み込み（radar_i, radar_q, tfm_ecg1等）
- `data/ecg_windowing.py`: 窓切り出し。**GDN0003のECGに41サンプル(20ms)のNaN欠損が
  実データに存在**し、素朴に窓全体をz-score正規化すると欠損が録音全体を汚染してNaN学習に
  なることが分かったため、`np.nanmean`/`np.nanstd`で正規化しつつ、欠損を含む窓は
  スキップする処理を入れた（テストで固定化済み: `tests/test_ecg.py`）
- `models/deep/ecg_cnn1d.py`: 全畳み込み1D CNN（dilation 1,2,4,8,16,32、`padding='same'`で
  入力と出力の長さを厳密に一致させるsequence-to-sequenceモデル）
- `training/ecg_trainer.py`: 損失はMSEだが、評価指標は窓内Pearson相関係数（振幅・位相のズレに
  頑健、波形の「形」が合っているかを見る）。`train.py`/`evaluate.py`の`family`振り分けに
  `"ecg_seq2seq"`を追加（既存の`deep`/`classical`はそのまま、1ファイル1責務を維持）
- Restingシナリオ、被験者7(train)/1(val)/2(test)、window_sec=4, stride_sec=2、50 epoch

### 結果

`configs/experiments/101_ecg_cnn1d_resting.yaml`、test_corr=**0.524**
（test被験者個別でもGDN0009=0.459、GDN0010=0.580と一貫）。train_corr=0.905まで到達し
val_corr(0.47〜0.51)との間に相応の過学習傾向があるが、**未知の被験者2名それぞれで
個別に有意な正の相関が出ている**点が、イヌHR/BRの結果（within_dog_corr -0.12〜0.17、
実質ゼロ）と決定的に異なる。

波形を可視化すると（[`reports/20260722_ecg_resting/waveform_example.png`](reports/20260722_ecg_resting/waveform_example.png)）、
心拍のタイミング（R波が出現する位置）はおおむね追えているが、QRS波特有の鋭い振幅・形状
（真値では振幅3〜4に達する鋭いスパイク）は再現できておらず、なだらかな山型に鈍っている。
`padding='same'`のみで受容野を広げるダウンサンプリングなしの畳み込みスタックは、鋭い遷移を
平滑化しやすい構造的な弱点を持つと考えられる。

### 含意・次に試すこと

- **「レーダから他センサの時系列波形を推定する」という方向自体は、少なくともヒトデータでは
  ゼロではない信号が確かに存在することを確認できた。** イヌのHR/BR予測で個体内相関が
  実質ゼロだったのとは対照的であり、目標の転換（ユーザ提案）は方向性として筋が良いと言える。
- ただし波形の**形**（QRS振幅・鋭さ）の再現度はまだ低く、健康モニタリングに使える水準
  （例えばR波タイミングからHRV/RR Intervalを高精度に逆算できる水準）には遠い。
  次のステップとして、(a) ダウンサンプリング+アップサンプリング構造（U-Net的）や
  周波数領域損失の導入、(b) Valsalva/TiltUp/TiltDownなど自律神経賦活シナリオでの検証
  （現在はRestingのみ）、(c) より多くの被験者（現在は10名中7/1/2、残り20名も取得可能）
  でのcross-validationを検討する。
- **イヌへの転用は、イヌ側にECG波形の正解データが存在しない限り不可能**（`data/raw/README.md`
  参照）。転用の前提として、イヌの拍単位データ取得（Polar H10等、manager-agent側の未解決事項）
  が必要になる。

### 理想的なMAEの目安

- **先行研究との比較**: 同一データセットの原著（Ahmed et al. 2024）は、**同一個体内**でのレーダ
  vs 参照センサ比較においてHR MAE 3.7 bpm・BR MAE 2.3 breaths/minを報告している
  （[`manager-agent/research/dog-mmwave-rri/summaries/03_Ahmed-2024_dog-uwb-public-dataset.md`](../manager-agent/research/dog-mmwave-rri/summaries/03_Ahmed-2024_dog-uwb-public-dataset.md)）。
  ただし本リポジトリは**未知個体への汎化**（leave-dogs-out）を課題にしており、同一個体内比較より
  本質的に難しい。単純に「3.7を切れば良い」とは言えない。
- **現実的な目標**: 上記のtrivial/oracleを使い、
  - HR: trivial=18.8を明確に下回る（目安として、013の17.5よりさらに踏み込んで**15以下**を
    最初のマイルストーンとし、Ahmedの同一個体内精度3.7 bpmを最終的な目安とする）。
    oracle=0.99が理論下限だが、未知個体への汎化ではここまでは近づけない可能性が高い。
  - BR: 現行のScenario1（麻酔下）のままでは目標設定自体が無意味
    （oracle=0.000＝学習すべき信号がほぼ無い）。**Scenario2（覚醒・自由行動、より変動の大きい
    条件）や自前データ取得で、犬ごとにBRが変動する条件に切り替えない限り、モデル比較として
    意味のある数値目標は立てられない。**
- **今後の全てのモデル比較で、`scripts/compute_baselines.py` の出力をtable末尾に併記し、
  trivial_maeを下回っているかを機械的にチェックする運用とする。**

## HR Transformer対照実験の結果（2026-07-22）

001（lr=1e-4, 100epoch）がtest MAE 31.0だった原因を切り分けるため、学習率・エポック数・
スケジューラを変えた4パターンを対照実験した（[`reports/20260722_hr_transformer_ablation/`](reports/20260722_hr_transformer_ablation/)）。

| 設定 | epoch | best_val_mae | final_train_mae | test_mae |
|---|---|---|---|---|
| 001 baseline (lr=1e-4) | 100 | 31.44 | 24.19 | 31.03 |
| 013 lr=5e-4 | 100 | 2.21 | 0.47 | **17.45** |
| 014 lr=1e-3 | 100 | 2.86 | 0.98 | 19.48 |
| 015 lr=1e-4（epochのみ300に延長） | 300 | 0.99 | 0.47 | 18.74 |
| 016 warmup(10ep)+cosine, peak lr=5e-4 | 200 | 1.76 | 0.30 | 17.83 |

（参考: 同じHRタスクの最良の古典MLは random_forest で test_mae=18.74）

**1. 「未収束」の診断は正しかった。** [`learning_curves.png`](reports/20260722_hr_transformer_ablation/learning_curves.png)
の通り、001（青線）はval MAEがほぼ動かず高止まりして見えるのに対し、lrを上げる（013・014）か
epochを延ばす（015）かスケジューラを使う（016）かのいずれでも、val MAEはepoch 100〜200前後で
1〜3程度まで下がる。001の失敗は単純に学習率が低すぎたことによる学習不足であり、
アーキテクチャの限界ではなかった。

**2. lr=5e-4（013）が最良で、古典MLを上回った。** test MAE 17.45は、これまでの
全12モデル中で最良の値（[`reports/20260722_baseline/table.md`](reports/20260722_baseline/table.md)参照）。
lr=1e-3（014）はやや不安定（学習曲線に大きめの振動があり、best_val_mae 2.86とやや高い）で
性能もやや劣る。過度に大きい学習率は収束を速めるが安定性を犠牲にする、という通常の傾向と一致する。

**3. しかし、より重大な問題が見つかった: best_val_mae と test_mae が大きく乖離している。**
013はbest_val_mae=2.21（bpm）まで下がっているのに、同じチェックポイントのtest_mae は17.45で
**8倍近い差**がある。015に至ってはbest_val_mae=0.99（1 bpm未満）まで下がりながらtest_mae=18.74。
train_mae も0.3〜1.0まで下がっている。これは「HRを予測する一般的な関数」を学習したというより、
**val犬（No8）1頭の個体特有のレーダ信号パターンを実質的に記憶した**ことを示唆する。
val split が1頭しかいないため、「val_mae最小のepochを選ぶ」というモデル選択の基準そのものが、
未知個体への汎化ではなくその1頭への適合度を測ってしまっている。
[`learning_curves.png`](reports/20260722_hr_transformer_ablation/learning_curves.png) の赤線
（015, 300epoch）も、epoch 230付近でval MAEが1台から5台へ再上昇しており、val 1頭への
過適合が進行している様子が見える。

**この問題は今回の比較全体（001〜016）の犬分割（train 7頭・val 1頭・test 2頭の固定1分割）に
共通する構造的な弱点であり、Transformer固有の問題ではない。** 全10頭という個体数の少なさに対し
単一の固定分割で「モデルAはモデルBより優れている」と結論するのは、val・testに割り当てられた
特定の3頭の個体差を見ているだけの可能性を否定できない。次にモデル比較をやり直す際は、
犬を入れ替えたleave-few-dogs-out cross-validation（例: 10頭を5foldに分け、毎foldでtrain/val/testの
犬を入れ替えて平均・分散を見る）を導入すべきである。

## 今後の拡張予定（研究計画の全体像）

現行実装は、深層モデル3種（Transformer/CNN1D/LSTM）と古典ML3種（Ridge/RandomForest/
GradientBoosting）によるHR/BR単一タスク回帰のベースライン比較である。
以下は卒業研究としての拡張方向（優先順位未確定）。

- **マルチタスク学習**: 心拍数・呼吸数を同一エンコーダから同時予測する。
  `VitalsTransformer` は `n_outputs` を複数に増やせば構造上は対応できるが、
  損失の重み付けと学習ダイナミクスの検証が必要。
- **複素領域モデル**: レーダの生信号（I/Q）を複素数のまま扱うモデル。
  現行の `RawData_No*.csv` が実数467列（既に何らかの前処理済みの可能性）である点を要確認。
  複素ネットワークを使うには、生I/Qデータへのアクセス可否から調査する必要がある。
- **超次元コンピューティング（HDC）**: Transformerとの比較対象、または軽量な代替アーキテクチャとして検討。
- **モデルの小型化**: 蒸留・枝刈り・量子化など。組込み実装を見据えた展開。
- **健康モニタリング**: 個体別ベースラインの確立や異常検知など、応用側のタスク設計。
- **RR Interval・ECG波形予測**: 犬データセット（Ahmed et al. 2024 シナリオ1）には
  拍単位の正解が存在しないため引き続き実装不可（`data/raw/README.md` 参照）。
  2026-07-25、拍単位の正解付きヒトmmWaveデータセットMMECG(Chen et al. 2022)を入手し、
  201番台として本実装を開始した（下記「MMECGでのRR Interval・ECG波形予測: 実装フェーズ」参照）。

## MMECGでのRR Interval・ECG波形予測: 実装フェーズ（2026-07-25）

ユーザから、まさに本題（RR Interval・ECG波形予測）にマッチするMMECGデータセット
（Chen et al. 2022、TI AWR1843 mmWave radar、拍単位ECG同期、35名・91トライアル）を
契約済み配布物として受け取り、`data/raw/MMECG202211.rar`に配置。展開・データ構造・
被験者数などの詳細は`data/raw/README.md`「MMECGデータセット」節を参照。

**重要な事実**: 91トライアルは11被験者（id: 1,2,5,9,10,13,14,16,17,29,30）に集約される
（1被験者2〜23トライアルと偏りが大きい）。トライアル単位ではなく被験者ID単位で
train/val/test分割する必要がある（犬データの教訓と同型。`data/mmecg.trial_ids_for_subjects`
で被験者ID→トライアルIDの変換を一元化）。

### 実装した内容（config `201`〜`215`、family `mmecg_*`、AGENTS.md参照）

ユーザの要望（NCP・複素領域モデル・GNN(+GAN、拍単位PQRSTグラフ)という3つの「本命」を、
通常のTransformer/CNN/LSTM/古典MLと比較する）に沿って、以下をすべて実装した
（単一split: train=[1,2,5,9,10,13,14], val=[16], test=[17,29,30]）。

| config | family | model | 役割 |
|---|---|---|---|
| 201/202 | `mmecg_seq2seq`/`mmecg_rpeak_seq2seq` | `ecg_cnn1d`/`rpeak_cnn1d` | CNNベースライン（既存101/102モデルを`in_channels=50`で流用） |
| 203/204 | 同上 | `ecg_transformer`/`rpeak_transformer` | Transformerベースライン（新規実装） |
| 205/206 | 同上 | `ecg_lstm`/`rpeak_lstm` | LSTMベースライン（新規実装） |
| 207/208 | 同上 | `ecg_ncp`/`rpeak_ncp` | **本命1: NCP**（`ncps.torch.CfC`+`AutoNCP`配線） |
| 209/210 | 同上 | `ecg_complex_cnn`/`rpeak_complex_cnn` | **本命2: 複素領域モデル**（Hilbert変換でanalytic signal化した入力を複素畳み込み+ModReLUで処理） |
| 211/212/213 | `mmecg_classical_rr` | ridge/random_forest/gradient_boosting | 古典MLベースライン（密波形出力に不向きなため、窓内平均RR Interval[ms]のscalar回帰に限定） |
| 214/215 | `mmecg_beatgraph`/`mmecg_beatgraph_gan` | `beatgraph_gnn` | **本命3: GNN(+GAN)**。R波中心のRCG segmentからPQRST 5点グラフ(時刻オフセット・振幅)を`torch_geometric.nn.GATConv`で回帰。215はさらに判別器との敵対的学習(LSGAN)を追加 |

**設計上の重要な注意**: GNN/GAN系統(214/215)は、CNN/Transformer/LSTM/NCP/複素CNN/古典MLとは
**異なる前提のタスク**である。後者6系統は「窓の中からR波をゼロから検出してRR Intervalを
求める」問題だが、GNN/GAN系統は「既知のR波位置を中心に、そのビートのPQRST形状を予測する」
問題であり、R波検出精度に依存しない。そのため両者を同じRR Interval MAEで直接比較すること
はできず、`scripts/compare_mmecg_models.py`でも別表として報告する。GNN/GANが答えているのは
「拍を検出できた後、その拍の詳細形状(PQRST)をどこまで再現できるか」という補完的な問いである。

正解のP/Q/S/T位置は`neurokit2.ecg_delineate`（method='peak'）による自動検出であり、
**臨床アノテーションではない疑似正解**である（ユーザ合意事項）。

### 実装上の技術的知見

- **NCPは1エポックあたり約5.5分と極めて遅い**（CfCの逐次時間展開によるもの、batch=32・
  seq_len=800・in_channels=50のGPU実測値）。CNN/Transformer/LSTM/複素CNNは同条件で
  1エポック数秒〜数十秒と桁違いに速い。NCPのみ`epochs`を50→15に落とした
  （他系統との比較時にこの学習量の差を必ず断り書きすること）。
- **PyTorch(2.5系)の`nn.Conv1d`は`dtype=torch.complex64`を指定するとそのまま複素畳み込み
  として動作する**（複素の重みを持つ通常の畳み込みとして自動的に複素乗算になる。逆伝播も
  動作確認済み）。一方`nn.BatchNorm1d`は複素dtypeで`NotImplementedError`になるため、
  複素領域モデルでは正規化層を使わず`ModReLU`（振幅にのみ非線形性をかけ位相を保存する
  活性化）のみで構成した。
- **`ncps.torch.CfC`のAPIは`CfC(input_size, units, ...)`で、`units`にWiringオブジェクト
  （`AutoNCP(units, output_dim)`）を渡すと自動的に疎結合配線モードになる**
  （`wiring`という名前のキーワード引数は存在しない）。`batch_first=True`で
  `(batch, seq_len, wiring.output_dim)`を返す。ncps==1.0.1で確認済み。

### 単一split結果（2026-07-25、全15config・201-215）

`scripts/compare_mmecg_models.py`による横断比較（[`reports/mmecg_comparison/comparison_full.json`](reports/mmecg_comparison/comparison_full.json)、全15run）。

**RR Interval MAE比較**（CNN/Transformer/LSTM/NCP/複素CNN/古典ML、R波検出→RR Interval算出という共通の物差し）:

| model | family | F1 | RR Interval MAE |
|---|---|---|---|
| rpeak_cnn1d | heatmap回帰 | 0.256 | **10.7ms**（全体最良） |
| rpeak_complex_cnn | heatmap回帰 | 0.415 | 18.5ms |
| ecg_complex_cnn | 波形回帰 | 0.373 | 19.4ms |
| rpeak_lstm | heatmap回帰 | 0.458 | 20.3ms |
| **ecg_ncp** | 波形回帰 | 0.311 | 21.1ms |
| ecg_cnn1d | 波形回帰 | 0.479 | 21.2ms |
| ecg_lstm | 波形回帰 | 0.350 | 21.5ms |
| ecg_transformer | 波形回帰 | 0.436 | 23.5ms |
| **rpeak_ncp** | heatmap回帰 | 0.054 | 30.0ms（深層モデル中最悪） |
| ridge | 古典ML(直接回帰) | N/A | 94.5ms |
| random_forest | 古典ML(直接回帰) | N/A | 95.4ms |
| gradient_boosting | 古典ML(直接回帰) | N/A | 96.9ms |
| rpeak_transformer | heatmap回帰 | 0.000 | N/A（未収束、ピーク検出ゼロ） |

**ビートグラフ(PQRST形状)比較**（GNN/GAN、既知R波前提の別タスク、直接比較不可）:

| model | time_mae | amp_mae |
|---|---|---|
| beatgraph_gnn (214) | **14.2ms**（最良） | 0.496 |
| beatgraph_gnn_gan (215) | 17.4ms | 0.481 |

### 解釈

- **101/102(Schellenberger, CW radar)と同じ傾向がmmWaveでも再現した: heatmap回帰(疎なイベント検出)が波形回帰(密な再構成)よりRR Interval精度で優れる。** 最良はrpeak_cnn1d(10.7ms)で、同じCNNバックボーンの波形回帰版(ecg_cnn1d, 21.2ms)の半分以下の誤差。101 vs 102の知見（EXPERIMENTS.md「101 vs 102のcross-validation」）と整合する結果が、入力モダリティを変えても支持された。
- **古典ML(RR Interval直接回帰)は深層モデル群に大差で劣る（94〜97ms vs 深層の10〜24ms）。** 犬HR/BRでは古典MLが深層モデルと同等以上だった（EXPERIMENTS.md「本実行から分かったこと」）のとは対照的。心拍帯域パワー最大の1チャネルに要約する特徴量設計では、50chの空間パターンや波形の微細構造を活かせず、直接RR Intervalを回帰するには情報が不足している可能性が高い。深層モデルは「まずR波位置を検出してから間隔を計算する」という間接的な定式化が有効に働いている。
- **rpeak_transformerが完全に未収束（F1=0）。** 犬HRでのTransformerの過去の失敗（学習率不足、EXPERIMENTS.md「HR Transformer対照実験の結果」）と同型の問題が再発した可能性が高い。lr/epoch数の対照実験は未実施（次の一手参照）。
- **複素領域モデル（本命2）は、同条件のCNN実数版と比べて明確な優位を示さなかった**（ecg_complex_cnn 19.4ms vs ecg_cnn1d 21.2msでわずかに上回るが、rpeak版は逆にcomplex(18.5ms)がrpeak_cnn1d(10.7ms)に負けている）。Hilbert変換で位相情報を追加しても、この単一splitでは一貫した優位性は確認できなかった。
- **GNN(本命3)のGAN拡張(215)は、MSE回帰のみ(214)より悪化した**（14.2ms→17.4ms）。判別器による正則化がこのタスク・データ規模ではむしろノイズになった可能性がある（adv_weight=0.1やGAN学習の安定化余地は残っている、次の一手参照）。
- **NCP（本命1）も明確な優位を示さなかった。** ecg_ncp(21.1ms)はecg_cnn1d(21.2ms)とほぼ同着で、
  rpeak_ncp(30.0ms)に至っては深層モデル中最悪だった。rpeak_ncpの学習曲線を見ると
  epoch10前後でtrain_corr・val_corrが0近辺に張り付いて動かなくなっており（出力がほぼ定数に
  潰れた退化）、`epochs`を50→15に落とした影響（学習不足）と、NCPというアーキテクチャ自体の
  限界を現時点では切り分けられていない。

### 総括: 3つの「本命」はこの単一split・単一設定では優位性を示さなかった

**NCP・複素領域モデル・GNN(+GAN)という3つの本命候補は、いずれもこの実験（単一split、
epochs=50が基本・NCPのみ15、ハイパーパラメータ探索なし）では、対応する通常DNNベースライン
（CNN/Transformer/LSTM）に対して明確な優位を示せなかった。** 全体最良はheatmap回帰CNN
(rpeak_cnn1d, 10.7ms)という「本命ではない」ベースラインであり、これは101/102(Schellenberger)
で確立した「疎なイベント検出として定式化する方が波形回帰より優れる」という知見（タスク定式化の
選択）が、モデルアーキテクチャの選択（本命かどうか）よりも支配的な要因だったことを示唆する。
ただし、これは「本命が劣る」ことの確定的な証拠ではなく、(a) 単一split・11被験者という
小規模設定での結果であり犬データと同様fold入れ替えで再現しない可能性がある、(b) 各本命モデルは
ハイパーパラメータ探索を一切行っていない素朴な初期実装である、という2点で結論を保留すべきである。

### 先行研究との比較（2026-07-25追記）

同一データセット(MMECG)の原著論文（Chen et al. 2022, RCG2ECG）は、**全35被験者・
35-fold leave-one-subject-out CVで、RR Interval中央値誤差わずか3ms（90percentile 9ms）**を
達成している。我々の最良値10.7ms（単一split・11被験者のみ）とは3倍以上の差があり、
その要因は「本命モデルかどうか」よりも(a)被験者数・評価方法（単一split vs 35-fold CV）、
(b)レーダ信号処理・アーキテクチャの作り込みの差にあると考えられる。詳細な比較表と解釈は
[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
を参照。**結論: MAE 10.7msはまだ改善余地が大きく、新奇アーキテクチャの追加より先に
Leave-Subjects-Out CVへの切り替えとデータ拡充・前処理の作り込みを優先すべき。**

### 次に試すこと

- 単一split結果は動作確認レベルであり、11被験者という少なさ（犬10頭の教訓と同型）を踏まえ、
  Leave-Subjects-Out cross-validation（`scripts/run_mmecg_cross_validation.py`は未実装、
  `scripts/run_dog_cross_validation.py`・`test_cv_significance.py`のパターンを踏襲して
  次段階で実装）と統計検定を行うまでは「どのモデルが優れているか」の結論を出さない。
  特にrpeak_cnn1dの10.7msという最良値は、犬データで再三見られた「単一分割の勝者は
  fold入れ替えで再現しない」パターンを疑うべきである。
- 上記の先行研究比較を踏まえ、モデルの追加実験よりも先に(a)MMECGの残り24被験者分の
  データ入手可否を確認する、(b)Chen et al.型の前処理（複数チャネルの重み付き統合等）や
  μ-law損失設計を取り入れるアブレーションを優先する。
- rpeak_transformerの未収束を学習率・スケジューラの対照実験で切り分ける（013と同じ手順）。
- GNN/GANの敵対的重み(adv_weight)やGAN学習の安定化（勾配ペナルティ等）を調整し、
  215が214を上回れるか再検証する。予測グラフと真のグラフの可視化による質的評価も追加する。
- 複素領域モデルが優位を示せなかった要因（Hilbert変換の位相情報がこのタスクに寄与しないのか、
  ModReLU中心の単純な構成が力不足なのか）をアブレーションで切り分ける。

## 精度追求フェーズ: 数学的根拠に基づく再設計（2026-07-25、config 216-224）

前節を受け、ユーザから「ここからは精度を本気で追求する段階」として、Transformer・CNN・
複素領域・NCP・GANの5軸を、アーキテクチャ・入力設計が複雑になってもよいので数学的根拠に
基づいて再設計してほしいとの依頼があった。追加の文献調査（radarODE, LifWavNet, RF2ESG/
Doppler-GAN論文, Efficient Edge-AI Models for ECG）を踏まえた再設計を実施し、
`configs/experiments/216`〜`224`として実装・本番実行した（詳細な設計根拠は
[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「追加調査」節を参照）。201-215（シンプル版）は変更せず据え置き、比較対象として残した。

### 結果（単一split、`reports/mmecg_comparison/comparison_v2_partial.json`）

| model | 系統 | F1 | RR Interval MAE | 備考 |
|---|---|---|---|---|
| rpeak_cnn1d (v1) | heatmap CNN | 0.256 | **10.7ms（全体最良、変わらず）** | 201-215のシンプル版が依然トップ |
| **ecg_unet1d** | 波形U-Net | 0.498 | **15.9ms** | **本フェーズの最大の成功**。波形回帰系統では圧倒的最良 |
| rpeak_complex_cnn (v1) | heatmap複素 | 0.415 | 18.5ms | |
| rpeak_complex_cnn_v2 | heatmap複素+ComplexBN | 0.530 | 18.5ms | F1は改善したがRR MAEは同着 |
| ecg_complex_cnn (v1) | 波形複素 | 0.373 | 19.4ms | |
| rpeak_lstm | heatmap LSTM | 0.458 | 20.3ms | |
| rpeak_unet1d | heatmap U-Net | 0.326 | 20.4ms | 波形版ほどの改善は見られず |
| ecg_ncp | 波形NCP | 0.311 | 21.1ms | ecg_cnn1dとほぼ同着（変わらず） |
| ecg_cnn1d (v1) | 波形CNN | 0.479 | 21.2ms | |
| ecg_lstm | 波形LSTM | 0.350 | 21.5ms | |
| ecg_conformer | 波形Conformer | 0.409 | 21.8ms | |
| rpeak_conformer | heatmap Conformer | 0.380 | 21.9ms | **rpeak_transformer(F1=0, 未収束)からは大幅改善** |
| ecg_complex_cnn_v2 | 波形複素+ComplexBN | 0.447 | 22.1ms | v1(19.4ms)よりむしろ悪化 |
| rpeak_unet1d(+GAN) | heatmap U-Net+PatchGAN | 0.389 | 22.3ms | 素のrpeak_unet1d(20.4ms)より悪化 |
| ecg_transformer (v1) | 波形Transformer | 0.436 | 23.5ms | |
| rpeak_ncp (v1, ep15) | heatmap NCP | 0.054 | 30.0ms | |
| ridge/RF/GB | 古典ML直接回帰 | N/A | 94.5-96.9ms | |
| rpeak_transformer (v1) | heatmap Transformer | 0.000 | N/A（未収束） | |

（GNN/GANのビートグラフ系統214/215は前節から変更なし: beatgraph_gnn 14.2ms、
beatgraph_gnn_gan 17.4ms）

### 解釈: 何が効いて、何が効かなかったか

- **フェーズD（U-Net、多重解像度）が唯一の明確な成功。** `ecg_unet1d`はecg_cnn1d(21.2ms)を
  約25%改善し(15.9ms)、波形回帰系統では圧倒的最良になった。LifWavNetが主張する「鋭いQRSと
  緩やかなP/T波を多重解像度で分離する」という設計思想が、簡略版（本式のlifting wavelet
  ではなく標準U-Net）でも実際に効くことを示す結果。一方`rpeak_unet1d`（heatmap版）は
  20.4msに留まり、波形回帰ほどの恩恵は見られなかった——heatmap自体が既に「疎なイベント」
  という単純な表現であり、多重解像度分離の恩恵が波形ほど大きくないためと考えられる。
- **フェーズB（Complex Batch Normalization）は検出率(F1)を改善したが、タイミング精度
  (RR MAE)は改善しなかった。** rpeak_complex_cnn_v2はF1を0.415→0.530まで押し上げたが、
  RR MAEはv1と同じ18.5msに留まった。これは「より多くの拍を検出できるようになったが、
  検出できた拍どうしの間隔精度は変わらなかった」ことを意味する。Complex BNによる学習安定化は
  実際に効いているが、タイミング精度というこのタスクの核心的な指標には届かなかった。
- **フェーズC（Conformer）は「未収束」問題は解決したが、精度自体は中位に留まった。**
  rpeak_transformerのF1=0（完全な未収束）という最悪の失敗は、rpeak_conformerでF1=0.380まで
  明確に解消された。局所畳み込みによる帰納バイアスが収束問題を解決するという仮説は支持された
  一方、最終的な精度は21.9msとトップ層には届かず、「収束させること」と「高精度を出すこと」は
  別の課題であることが分かった。
- **フェーズE（heatmap-GAN）は、素の生成器単体より悪化した（22.3ms vs rpeak_unet1dの
  20.4ms）。これはビートグラフGAN(215, 17.4ms vs 214, 14.2ms)と合わせて、密・疎どちらの
  表現でもGANの敵対的損失がこのタスク・データ規模ではむしろ悪影響という、2つの異なる
  ターゲット表現にまたがる一貫した否定的結果になった。** 先行研究（RF2ESG等）のGAN成功例は、
  より大きな訓練データ・より作り込まれた条件付け機構を持つ可能性が高く、本実装の
  train7被験者という小規模設定ではGANの分布学習に必要なサンプル数が不足している可能性が
  ある。
- **NCPは「タスクによって明暗が分かれる」という、当初の仮説を裏付ける結果になった。**
  ecg_ncp（滑らかな連続波形を出力）はecg_cnn1dと一貫して同等の精度を保つ一方、
  rpeak_ncp（鋭く疎なheatmapスパイクを出力）はlr調整・epochs延長（15→50、`223`）を
  行っても学習が定数出力へ収束せず崩壊したままだった（`/tmp/rpeak_ncp_v2.log`で
  epoch25時点でもtrain_corr=0.000のまま停滞を確認）。LTC/CfCニューロンの連続時間
  ダイナミクスは本質的にローパス的（緩やかな時定数で状態を積分する）ため、σ=10ms
  （fs=200Hzで2サンプル程度）という鋭いスパイクの生成に構造的に不向きという仮説が、
  ハイパーパラメータを変えても崩壊が再現したことで補強された。

### NCPの効率性（フェーズA、`reports/mmecg_comparison/efficiency_benchmark.json`・`efficiency_pareto.png`）

`ecg_ncp`は**24,981パラメータ（ecg_cnn1dの約14分の1）、8.0M MACs（同約36分の1）**と、
FLOPs・パラメータ数では他モデルを圧倒する。一方、CPU上でのバッチ1件あたり推論レイテンシは
**142.7ms（ecg_cnn1dの約27倍、全モデル中最悪）**であり、`ncps`ライブラリのCfC実装が
時間方向に逐次展開される（LSTMのようなcuDNN融合カーネルを持たない）ため、パラメータ数の
少なさがそのまま実行速度には結びついていない。**「Efficient Edge-AI Models for ECG」
(PMC, 2024)がSTM32上でLTC/CfCの省メモリ性を実証したのは専用の組込み実装によるものであり、
本実装（素朴なPyTorchでのCPU実行）ではFLOPs効率の良さがレイテンシに直結しないという、
誠実に報告すべき重要な限界がある。** パラメータ数・FLOPs・省メモリ性を活かすには、
量子化・ONNX/TFLite変換・専用ハードウェア（Loihi-2等）への展開が前提となる。

### 総括（精度追求フェーズ）

**5軸のうち明確な成功と呼べるのはフェーズD（U-Net、波形回帰限定）のみだった。**
Complex BN・Conformerは部分的な改善（検出率向上、未収束の解消）に留まり、GANは2種類の
ターゲット表現の両方で悪化した。NCPは波形タスクでは元々の強み（同精度・大幅な小型化）を
維持したが、heatmapタスクでは崩壊が確定的となった。**全体最良は依然としてrpeak_cnn1d
（v1、シンプルなheatmap回帰CNN、10.7ms）であり、「アーキテクチャを複雑にすれば精度が
上がる」という単純な仮説はこのデータ規模（単一split、train7被験者）では支持されなかった。**
これは、Chen et al.(2022)が全35被験者・35-fold CVという桁違いのデータ規模で3msを
達成している事実と合わせて読むと、**このタスクのボトルネックはモデルの表現力よりも
訓練データ量である可能性が高い**ことを示唆する。次段階でのLeave-Subjects-Out CV
（11被験者全体を使う）や、残り24被験者分のデータ入手が、モデル側の工夫よりも優先度が
高いという結論を補強する。

### 次に試すこと（精度追求フェーズ後）

- `ecg_unet1d`（本フェーズの最有力候補）を軸に、Leave-Subjects-Out CVで単一split結果が
  再現するかを最優先で検証する。
- NCP v2（223, epochs=50, lr=1e-4）の完走を待ち、224（ecg_ncp v2, epochs=50）の結果と
  合わせて本表を更新する。rpeak側の崩壊は確定的だが、ecg側がepochs=50でさらに改善するかは
  未確認。
- heatmap-GANの敵対的重み(adv_weight=0.1)を大幅に下げる、または判別器の学習率を下げるなど
  GAN学習の安定化を試したうえで、それでも改善しなければ「このタスク・データ規模では
  GANは不適」という結論を確定させる。

## NCPを差別化軸で評価する（2026-07-27、config 225-230）

ユーザから、NCPを論文の中心に据える場合は精度の横並び比較ではなく、NCP/LTC(Liquid Neural
Networks)が文献で報告する「精度競争とは別の強み」で評価すべきとの提案があった。文献調査
（`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「NCPを別軸で評価する」節）で
確認した2つの主張——(a) 同条件のノイズ下でRNN/CNNより性能劣化が緩やか、(b) 少データ・
ノイズが多い状況でも高い汎化性を保つ——を、本データセットで直接検証した。

まず224（ecg_ncp、停電で中断した`epochs=15→50`の再学習）を完走させ、test_corr=0.113
（15epoch版の0.115とほぼ同値）を確認。epoch数を増やしても大きな変化はなく、15epochの
時点で既に収束に近かったことが分かった。

### 実験1: ノイズ頑健性（`scripts/evaluate_noise_robustness.py`）

学習済みのecg_ncp・ecg_cnn1d・ecg_unet1dに、テスト時のRCG入力へガウスノイズを段階的に
注入し（noise_std=0〜2.0、信号のz-score標準偏差=1に対する相対値）、相関係数の劣化を比較した
（[`reports/mmecg_comparison/noise_robustness.png`](reports/mmecg_comparison/noise_robustness.png)）。

| noise_std | NCP | CNN | U-Net |
|---|---|---|---|
| 0.00 | 0.115 | 0.224 | 0.209 |
| 0.50 | 0.071 | 0.203 | 0.204 |
| 1.00 | 0.028 | 0.153 | 0.170 |
| 2.00 | **0.009** | 0.075 | 0.086 |

**結果は文献の主張と正反対だった。** NCPはノイズ0の時点で既に他モデルより低く、
noise_std=2.0では相関がほぼ消失(0.009)する一方、CNN・U-Netは同条件で0.075〜0.086を
維持した。相対的な劣化率（noise=2.0の相関 ÷ noise=0の相関）で見ても、NCP=7.4%、
CNN=33.6%、U-Net=41.1%であり、**NCPが最も脆弱**という結果になった。

### 実験2: データ効率（train被験者数2/4/7名で再学習、config 225-230）

train被験者数を7→4→2と減らし、ecg_ncp・ecg_cnn1d・ecg_unet1dを再学習して
test相関の劣化カーブを比較した
（[`reports/mmecg_comparison/data_efficiency.png`](reports/mmecg_comparison/data_efficiency.png)）。

| train被験者数 | NCP | CNN | U-Net |
|---|---|---|---|
| 2 | **-0.022**（負の相関） | 0.080 | 0.013 |
| 4 | 0.038 | 0.184 | 0.104 |
| 7 | 0.115 | 0.224 | 0.209 |

**こちらも文献の主張と正反対だった。** train被験者2名では、NCPだけが負の相関（=デタラメな
出力よりも悪い）に陥り、CNNは同条件でも0.080というプラスの信号を維持した。「少データでの
汎化性」という主張は、少なくとも本タスク・本実装では再現しなかった。

### 解釈

**NCP/LTCの文献における強み（ノイズ頑健性・少データでの汎化性）は、画像分類や制御タスクなど
別のドメインでの知見であり、本タスク（mmWaveレーダ→ECG波形回帰）にそのまま転移するとは
限らないことが、2つの独立した実験で一貫して示された。** むしろ本タスクでは、CNNの畳み込みが
持つ平行移動不変性という帰納バイアスの方が、少データ・高ノイズ環境での頑健性に寄与している
可能性が高い。NCPが実際に示した強み（`AutoNCP`配線による疎な構造、フェーズAで確認した
パラメータ数24,981・8.0M MACsという圧倒的な小ささ）は、標準的な精度・ノイズ・データ量の
条件下では発揮されず、**「同精度をクリーンな十分量のデータ条件下で、桁違いに少ないパラメータ
数で達成する」という一点に絞って主張するのが誠実である。**

### 結論: 論文でNCPを中心に据える場合の推奨

- **ノイズ頑健性・少データ汎化性を主張の軸にするのは推奨しない**（本実験で反証された）。
- **推奨する軸は「パラメータ効率」に一本化すること。** ecg_ncp(24,981 params)は
  ecg_cnn1d(356,417 params)の約14分の1のパラメータ数、8.0M MACs(約36分の1)で、
  クリーンな条件・train7被験者という十分なデータ量の下では同等の精度(0.113〜0.115)を
  達成している。この「同精度・大幅小型化」こそが、本実験で一貫して再現された唯一の
  NCPの強みである。
- **CPU推論レイテンシがボトルネックである点は必ず併記すること**（フェーズAで確認済み、
  142.7ms、CNNの27倍）。素朴なPyTorch実装ではパラメータ効率がレイテンシに直結しないため、
  「低消費電力」を主張するには量子化・ONNX変換・専用ランタイムへの展開が前提となる、
  という限界を正直に述べるべきである。
- 精度そのものを追求する場合の推奨構成は変わらず**rpeak_cnn1d**（RR Interval検出、全体
  最良）または**ecg_unet1d**（波形回帰、多重解像度分離が有効）であり、NCPはこれらの
  代替ではなく「同等の精度をより小さいモデルで達成する」という補完的な位置づけで報告する
  のが適切である。

## Chen et al.型アーキテクチャの導入とNCP進化系（2026-07-27、config 231-234）

ユーザから、(1) rpeak_cnn1d(10.7ms)とChen et al.(3ms)の差を、原著論文のアーキテクチャを
模倣する形で縮められないか、(2) NCPについても単純な素朴実装から進化させ、本データセットで
最も面白い結果を出せる構成を探索してほしい、との依頼があった。

### 空間融合モデル（Chen et al. 2022の発想を導入、config 231/232）

Chen et al.の原著論文（本文PDFを取得済み、`reports/mmecg_comparison/prior_work_accuracy_comparison.md`
参照）を読み直すと、そのアーキテクチャ（Fig.6）は50点のRCG計測を単なる畳み込みチャネルとして
扱うのではなく、**各点の3D位置(posXYZ)をTransformerの位置埋め込みとして時間特徴と融合する**
「空間融合」という設計を取っていることが分かった。**本リポジトリの201-224フェーズは、
posXYZを一度も読み込んでおらず完全に未使用だった**（`data/mmecg.py`の`MMECGRecording`に
フィールドすら無かった）。

この欠落を埋めるべく、`MMECGRecording`に`posxyz`フィールドを追加し（トライアルごとに
異なる値を持つことをposXYZ実データで確認済み）、`data/mmecg_spatial_dataset.py`
（RCG窓・posXYZ・ECG/heatmap窓を束ねる新Dataset）と`models/deep/ecg_spatial_fusion.py`/
`rpeak_spatial_fusion.py`（各点を重み共有1D convで時間エンコードし、posXYZの線形埋め込みを
加算後、時刻ごとに50点間でSelf-Attentionにより空間融合し、転置畳み込みで時間解像度を戻す
アーキテクチャ）を新設した。family=`mmecg_spatial_seq2seq`（2引数`forward(rcg, posxyz)`を
取るため既存の`ecg_registry.py`とは別の小さなレジストリを`training/spatial_fusion_trainer.py`
内に持つ）。

**結果**: `rpeak_spatial_fusion`がF1=0.545、`ecg_spatial_fusion`がF1=0.552と、
**これまで試した全モデル中で最高のF1（拍検出率）を達成した**（次点はrpeak_complex_cnn_v2の
0.530）。RR Interval MAEはrpeak_spatial_fusion=16.2ms、ecg_spatial_fusion=16.7msで、
全体最良のrpeak_cnn1d(10.7ms)には届かなかったが、201-224フェーズの大半のモデルより
優れた水準（前フェーズの2番手ecg_unet1dの15.9msに迫る）。**posXYZという未活用だった
情報を導入したことが、少なくとも「拍を検出できるかどうか」の面で明確な改善をもたらした**
——ただしタイミング精度(RR MAE)では、依然としてシンプルなrpeak_cnn1dの精密さに一歩譲る、
という複合的な結果になった。

### NCP進化系: Conv-NCP（局所畳み込み前処理+CfC、config 233/234）

Conformerが素のTransformerの未収束問題（rpeak_transformer, F1=0）を局所畳み込みの帰納バイアス
で解消したのと同じ発想を、NCPに適用した。`models/deep/ecg_conv_ncp.py`/`rpeak_conv_ncp.py`は、
QRS幅程度の受容野を持つ局所dilated conv（`ecg_conformer.py`の`ConvSubsampler`を再利用）で
RCGを前処理してから`ncps.torch.CfC`へ渡し、さらに`mixed_memory=True`（LSTM様のメモリセルで
CfCを補強するncpsのオプション）を有効にした2層CfCスタックとした。狙いは、(a) ノイズ頑健性
実験で判明したNCPの弱点（生信号のノイズに敏感）を畳み込みの平滑化で補うこと、(b) rpeak_ncpの
崩壊（heatmapの鋭いスパイクへの構造的不向き）を、より情報量の多い局所特徴を与えることで
緩和できるかを検証することだった。

2層CfCスタックは計算コストが単層の`ecg_ncp`よりさらに重く（1バッチ(32件)あたり
forward+backward計 約5.4秒、1epoch約10分）、`epochs`は50→20に調整して実行した
（結果は次回セッションで追記）。

### 次に試すこと

- Conv-NCP(233/234)の結果を追記し、局所畳み込み前処理がNCPの弱点（ノイズ感度・heatmap崩壊）を
  実際に緩和したかを検証する。
- 空間融合モデルのRR MAEとF1の乖離（検出率は最高だがタイミング精度は僅かに劣る）の原因を
  切り分ける。時間方向の重み共有Self-Attention（本実装の簡略化点）を、真のtime-varying
  attentionに置き換えると改善するか等。
- Chen et al.のもう一つの工夫であるμ-law companding変換+カテゴリカル損失、TCN自己回帰
  デコーダは今回未実装（時間予算の制約）。次段階の候補として残す。

## 空間座標を使ったGNN・GANの探索（2026-07-27、config 235-237）

ユーザから「空間座標を使うならGNN/GANの構成比較・アーキテクチャ探索もできるのでは」との
提案があり、mmWave point cloud向けのGNN研究（MMPoint-GNN, Gong et al. 2021、mmGAT等）を
追加調査した上で2つの発展形を実装した。

### 空間GNN（実距離k近傍グラフ+GATConv、config 235/236）

`ecg_spatial_fusion`（231/232、50点全結合のSelf-Attentionで空間融合）は「どの点も等しく
参照しあう」近似であり、実際の3D距離構造を使っていなかった。`models/deep/ecg_spatial_gnn.py`/
`rpeak_spatial_gnn.py`は、posXYZの実距離から構築したk近傍グラフ（k=6、`torch_geometric`の
`knn_graph`は`pyg-lib`依存のため自前実装、`models/deep/spatial_graph.py`）上で`GATConv`に
よるメッセージパッシングを行う、より物理的に妥当な設計にした。

**結果は本セッション最大の成果の一つとなった。** `rpeak_spatial_gnn`は**RR Interval MAE
12.6ms、F1=0.511**を達成し、**rpeak_cnn1d(10.7ms)に次ぐ全体2位**（前フェーズの
`ecg_spatial_fusion`の16.2msを明確に上回る）。しかもパラメータ数は33,985と
`ecg_spatial_fusion`(48,833)より少ない。**「全結合Self-Attention」から「実距離に基づく
k近傍グラフ」へ変更しただけで、より少ないパラメータでより高い精度を達成した**——
これは、空間融合という設計思想そのものだけでなく、「どうグラフ構造を与えるか」という
帰納バイアスの妥当性が精度に直結することを示す、説得力のある結果である。

### GAN×空間モデル（config 237）

ユーザの提案「空間座標をGANにも使ってみては」を受け、これまで最高のF1を出した
`rpeak_spatial_fusion`(232)を生成器としたheatmap敵対的リファインメント
（`training/spatial_heatmap_gan_trainer.py`、222と同じPatchGAN判別器）を試した。

**結果はF1=0.379・RR MAE=20.9msで、GANなし版(232: F1=0.545, 16.2ms)より明確に悪化した。**
これで、PQRSTグラフ(215)・heatmap+U-Net(222)・heatmap+空間融合(237)という**3種類の異なる
生成器・2種類の異なるターゲット表現すべてでGANが悪化させる**という、一貫した否定的結果が
出そろった。本データセットの規模（train7被験者）では、生成器の種類やターゲット表現に
関わらずGANの敵対的損失が正則化として機能しないことが、繰り返し実証されたと言える。

### 現時点のRR Interval MAEランキング（全モデル中）

| 順位 | model | RR MAE | 備考 |
|---|---|---|---|
| 1 | rpeak_cnn1d (v1) | 10.7ms | シンプルなheatmap回帰CNN |
| **2** | **rpeak_spatial_gnn** | **12.6ms** | posXYZ実距離k近傍グラフ+GATConv（本セッション成果） |
| 3 | ecg_unet1d | 15.9ms | 多重解像度U-Net（波形回帰） |
| 4 | rpeak_spatial_fusion | 16.2ms | posXYZ全結合Self-Attention |
| 5 | ecg_spatial_fusion | 16.7ms | 同上（波形回帰版） |
| 6 | ecg_spatial_gnn | 18.0ms | 同上のGNN版（波形回帰版、heatmap版ほどの恩恵なし） |

### Conv-NCPの結果（233/234完走、崩壊は明確に解消した）

| model | F1 | RR MAE | 参考: 対応するv1 |
|---|---|---|---|
| ecg_conv_ncp | 0.400 | 19.5ms | ecg_ncp: F1=0.311, 21.1ms |
| rpeak_conv_ncp | 0.326 | 18.8ms | **rpeak_ncp: F1=0.054, 30.0ms（学習が完全崩壊）** |

**局所畳み込み前処理+mixed_memoryという進化が、狙い通りrpeak_ncpの崩壊を明確に解消した。**
rpeak_ncp(v1)はBCE損失が学習中に定数付近で停滞し（corr≈0で50epoch中ずっと停滞）、
「heatmapの鋭いスパイクにLTC/CfCの連続時間ダイナミクスが構造的に不向き」という仮説を
裏付けていたが、`rpeak_conv_ncp`はF1=0.326・RR MAE=18.8msという、崩壊とは程遠い
実用的な水準まで改善した。ecg側も21.1ms→19.5msとわずかに改善している。**「NCPに
渡す前に局所的な形状特徴を抽出しておく」という設計変更（Conformerがrpeak_transformerの
未収束を解消したのと同じ発想）は、NCP系統でも明確に有効だった。**

ただし絶対水準では依然としてrpeak_spatial_gnn(12.6ms)やrpeak_cnn1d(10.7ms)には届いておらず、
「NCPの構造的弱点を緩和した」レベルであり「NCPが最良になった」わけではない点には留意が必要。

### 最終ランキング（本セッション全体、単一split・RR Interval MAE）

| 順位 | model | RR MAE | 系統 |
|---|---|---|---|
| 1 | rpeak_cnn1d | 10.7ms | シンプルなheatmap CNN(201-215フェーズ) |
| **2** | **rpeak_spatial_gnn** | **12.6ms** | posXYZ実距離k近傍グラフ+GATConv（本セッション最大の成果） |
| 3 | ecg_unet1d | 15.9ms | 多重解像度U-Net |
| 4 | rpeak_spatial_fusion | 16.2ms | posXYZ全結合Self-Attention |
| 5 | ecg_spatial_fusion | 16.7ms | 同上（波形回帰版） |
| 6 | ecg_spatial_gnn | 18.0ms | k近傍グラフGNN（波形回帰版） |
| 7 | rpeak_conv_ncp | 18.8ms | **局所畳み込み+CfC（NCP進化系、v1の30.0msから大幅改善）** |
| 8 | rpeak_complex_cnn/v2 | 18.5ms | 複素領域CNN |
| 9 | ecg_complex_cnn | 19.4ms | 同上（波形回帰版） |
| 10 | ecg_conv_ncp | 19.5ms | NCP進化系（波形回帰版、v1の21.1msから改善） |
| … | rpeak_spatial_fusion+GAN | 20.9ms | GAN適用で悪化(3例目) |
| … | ecg_ncp (v1) | 21.1ms | シンプルなNCP |
| … | rpeak_ncp (v1) | 30.0ms | 崩壊（Conv-NCPで解消済み） |

### 総括: このセッションで確立した3つの明確な知見

1. **posXYZ(空間座標)の活用は本物の改善をもたらした。** ただし「どう使うか」が重要で、
   全結合Self-Attention（空間融合）よりも実距離に基づくk近傍グラフ+GNN（空間GNN）の方が、
   パラメータ数を減らしながら精度も上回った。物理的に妥当な帰納バイアス（近い点同士が
   相互作用する）を明示的に与えることの価値を示す、本セッションで最も説得力のある結果。
2. **GANは一貫して悪化させる。** PQRSTグラフ・heatmap+U-Net・heatmap+空間融合という
   3種類の生成器・2種類のターゲット表現すべてで敵対的損失が性能を落とした。train7被験者
   という規模ではGANの分布学習が機能しないという結論を、独立した3実験で再確認した。
3. **NCPの構造的弱点（heatmapの鋭いスパイクへの不向き）は、局所畳み込み前処理で明確に
   緩和できる。** rpeak_ncpの完全崩壊(F1=0.054)がConv-NCPでF1=0.326まで改善し、NCPを
   「素朴に使うと弱いが、CNN的な前処理と組み合わせれば実用域に届く」モデルへと押し上げた。
   ただし最良モデル(rpeak_spatial_gnn, rpeak_cnn1d)には届かず、NCPの主張軸は依然として
   フェーズAで確立した「パラメータ効率」に置くのが最も誠実である。

### 次に試すこと

- rpeak_spatial_gnnがなぜecg_spatial_gnnより優れるか（heatmap定式化との相性が良いのか）を、
  101/102・201/202で確認済みの「疎なイベント検出が波形回帰より優れる」傾向と合わせて整理する。
- k_neighbors(現在k=6)やGATConvの層数を変えたアブレーションで、rpeak_spatial_gnnの
  12.6msをrpeak_cnn1dの10.7msに近づけられるか、あるいは上回れるかを検証する。
- rpeak_cnn1dのバックボーン（dilated conv）にposXYZを組み込んだハイブリッド
  （空間GNN由来の点別エンコーダ+rpeak_cnn1dの検出ヘッド）を試し、10.7msをさらに更新できるか
  検証する。
- 単一split結果はここまですべて動作確認レベルである。11被験者という少なさを踏まえ、
  Leave-Subjects-Out CVで上位モデル（rpeak_cnn1d, rpeak_spatial_gnn, ecg_unet1d）の
  順位が再現するかを次段階で必ず検証する。

## 波形推定モデル(ecg_*系)だけの比較: 波形重ね描き+正式な相関ランキング（2026-07-27）

ユーザから、波形推定タスク(heatmap回帰のrpeak_*系やGNN/GAN系は対象外、密なECG波形を
直接出力するecg_*系11モデルのみ)に絞って、推定波形と正解波形を重ね描きしたグラフと、
波形相関の比較をしてほしいとの依頼があった。`scripts/plot_waveform_comparison.py`を新設し、
[`reports/mmecg_comparison/waveform_prediction_grid.png`](reports/mmecg_comparison/waveform_prediction_grid.png)
（11モデル×同一の例示窓での重ね描き）と
[`reports/mmecg_comparison/waveform_correlation_ranking.png`](reports/mmecg_comparison/waveform_correlation_ranking.png)
（test被験者全トライアル・全窓での正式な平均相関ランキング）を作成した。

**実装中に重大なバグを発見・修正した**: 当初、例示用のトライアルIDとして「17」を
決め打ちで使ったが、`data/mmecg.py`のトライアルID(.matファイル番号)と被験者IDは別の
名前空間であり、「17.mat」は実際には**訓練被験者(id=9)**に属するトライアルだった
（被験者ID=17のトライアルはファイル48〜56）。この状態でtest用チェックポイントを評価すると
相関が0.5〜0.99という異常に高い値になり（モデルが学習時に見たデータだったため）、
バグに気づくきっかけになった。修正後は、`trial_ids_for_subjects(raw_root, test_subjects)`で
実際のtestトライアルIDを取得し、それがtest集合に含まれることをassertで検証するようにした
（`scripts/plot_waveform_comparison.py`のコメントに経緯を明記）。

### 波形相関ランキング（test被験者全体、正式な平均）

| 順位 | model | 相関係数 |
|---|---|---|
| 1 | ecg_cnn1d | 0.224 |
| 2 | ecg_conv_ncp | 0.210 |
| 3 | ecg_unet1d | 0.209 |
| 4 | ecg_transformer | 0.205 |
| 5 | ecg_spatial_gnn | 0.188 |
| 6 | ecg_complex_cnn | 0.171 |
| 7 | ecg_complex_cnn_v2 | 0.151 |
| 8 | ecg_spatial_fusion | 0.150 |
| 9 | ecg_conformer | 0.135 |
| 10 | ecg_lstm | 0.134 |
| 11 | ecg_ncp | 0.113 |

**波形の「形」を再現する精度（相関）で見ると、シンプルなecg_cnn1dが依然最良で、
本セッションで再設計した各モデルはいずれも上回れなかった。** ただし2位はConv-NCP
（NCP進化系）であり、素のecg_ncp(11位, 0.113)から大幅に改善している——これは
RR Interval MAEでの改善（21.1ms→19.5ms）と整合する結果であり、「局所畳み込み前処理が
NCPの弱点を緩和する」という知見を、波形相関という独立した指標からも裏付けた。

一方、RR Interval MAEでは2位だった`rpeak_spatial_gnn`に対応する`ecg_spatial_gnn`は、
波形相関では5位(0.188)に留まる。これは「疎なイベント検出（heatmap）に強いアーキテクチャが、
必ずしも密な波形再構成にも強いとは限らない」ことを示しており、101/102・201/202以来
繰り返し確認してきた「タスク定式化とアーキテクチャの相性」というテーマの一貫性を裏付ける。

### 波形の質的な観察（重ね描き図より）

どのモデルもR波のタイミング（鋭いスパイクの位置）はおおむね捉えているが、**QRSの鋭い振幅
（正解は5.3付近まで達するが、予測はどのモデルも3〜4程度に鈍る）を再現できているモデルは
一つもない。** これは2026-07-22のヒトECG手法検証（Schellenbergerデータセット、
`padding='same'`のみのdilated convスタックは鋭い遷移を平滑化しやすい）で確認した限界と
同じ構造の問題であり、mmWaveデータでも同様に残っている。振幅の鈍りに対する対策
（例えば振幅に重みを掛けた損失関数、Chen et al.のμ-law変換等）は依然として未着手であり、
次の一手として優先度が高い。

## 相関0.90再現の第一歩: min-max正規化と評価粒度の検証（2026-07-28）

詳細な数値・解釈は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「相関0.90再現の第一歩」節に記載。要点のみここに残す。

- `data/mmecg_windowing.py`に`normalization: "zscore"|"minmax"`を追加し、201と正規化方式
  だけが異なる対照実験240を実行。**min-max([-1,1])はz-scoreとほぼ同水準**（窓内相関0.212 vs
  0.224）で、改善効果は確認できなかった。学習曲線もむしろval_corrの振動が大きくなった。
- `scripts/evaluate_per_beat_correlation.py`でChen et al.と同じ評価粒度（拍単位セグメント化
  後にPearson相関、中央値報告）に揃えて201・240を再評価。中央値は0.16〜0.22に留まり、
  **評価方法を揃えても0.90との差はほとんど埋まらなかった**。0.224 vs 0.90の差は評価指標の
  計算方法の違いではなく、実質的なモデル性能の差であることが確認された。
- 残る最有力な未検証要素は**TCN自己回帰デコーダ**（過去の実ECGサンプルをデコーダ入力に
  含めるteacher forcing構成）と**μ-law companding+カテゴリカル損失**。これらは正規化方式や
  評価方法と異なり本セッションでは未実装のままであり、次の実装対象として優先度が高い。

## Chen et al.アーキテクチャの忠実な再現を実装・実行した結果: 再現できず、原因を特定（2026-07-28）

詳細な数値・解釈は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「Chen et al.アーキテクチャの忠実な再現を実装・実行した結果」節に記載。要点のみここに残す。

- 上記「次の一手」で残っていたTCN自己回帰デコーダ+μ-law companding+256値カテゴリカル損失を
  `models/deep/chen2022_reconstructor.py`（空間時間エンコーダ+TCN自己回帰デコーダ）、
  `data/mulaw.py`、`training/chen2022_trainer.py`として実装し、
  `configs/experiments/241_mmecg_chen2022_repro.yaml`で学習・評価した。
- 教師強制での学習は順調に収束したが、**原著と同じ自己回帰生成で評価すると窓内相関0.065・
  拍単位相関中央値0.014となり、既存の素朴なfeedforward回帰(201: 0.224/0.163)より明確に
  悪化した。**
- 原因を診断した結果、**teacher forcing下の高精度(相関0.996)は、RCGを無視して「1つ前の
  正解サンプルをそのままコピーする」だけのtrivial baseline(相関0.977)とほぼ同じ値であり、
  ECG自身の強い自己相関から来る見かけの精度**であることが判明した。自己回帰生成では
  このショートカットが使えず、誤差が蓄積して波形が停滞・崩壊する（exposure bias）。
- **結論: 現状の訓練データ規模(7被験者)では、アーキテクチャを原著に近づけても0.90は
  再現できない。** 自己回帰デコーダは条件付け特徴が弱いとfeedforward回帰より悪化させる
  諸刃の剣であり、原著が0.90を達成できているのは34人分の訓練データ・より作り込まれた
  信号処理段による条件付け特徴の強さに起因する可能性が高い。「訓練データ量が支配的な
  ボトルネック」という本報告書の一貫した仮説が、最後の未検証アーキテクチャ要素を
  潰したことでさらに強く裏付けられた。
- **今後の教訓**: 自己回帰系モデルを評価する際は、必ず「trivial copy-prev-sample
  baseline」を並記し、teacher forcing下の高精度がショートカットによるものでないか
  毎回検証すること。

## データ規模の再検証: 「11被験者では再現不可能」という結論は誤りだった（2026-07-28）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「データ規模の再検証」節、および[`data/raw/README.md`](data/raw/README.md)「`MMECG202211.rar`は
原著者配布物の「全体」であり、取得漏れではない」節に記載。要点のみここに残す。

- ユーザの指摘を受け、`MMECG202211.rar`を`unrar lb`で一覧確認したところ隠れた追加データは
  無く、既に展開済みの91トライアル・11被験者が**原著者配布物の全量**であることを確認した。
  原著者のGitHub([jinbochen0823/RCG2ECG](https://github.com/jinbochen0823/RCG2ECG))には
  「4.55時間のみ公開済み、残り(約24被験者分)は認可プロセス中」との明記があり、実測した
  録音時間269.2分(4.49時間)とほぼ一致する。**残りのデータは機関印付き同意書を
  `jinbochen@mail.ustc.edu.cn`へ送付すれば入手できる経路が存在する**（人間側の作業が必要
  なため本セッションでは未実施、次の一手として提案）。
- Chen et al.とは無関係の研究室(Xi'an Jiaotong-Liverpool University・HKUST(GZ))による
  **radarODE**(arXiv:2408.01672)が、**我々と全く同じ91トライアル・11被験者**を使い、
  11-fold leave-one-subject-out CVで評価していることを本文精読で確認した。同論文の
  Table VIによれば、**Chen et al.のアーキテクチャの再現でPCC(相関)87.9%、radarODE自身で
  92.6%**を、このデータだけで達成している。
- **これは前回セッションの結論「11被験者では原著の0.90は再現不可能」を覆す。** 真の
  ボトルネックはデータ量ではなく方法論だった。radarODEとの実装差は主に: (a) 生時系列RCG
  ではなくSynchrosqueezed Wavelet Transform(SST)による時間周波数領域への変換、(b) 複数拍を
  含む窓を一括回帰するのではなく、50チャネル多数決によるPPI(拍間隔)推定で1心拍単位に
  切り出してから再構成するタスク設計、(c) ECGの力学モデル(ODE、PQRST各波をガウシアン
  パルスとして表現)による振幅事前分布、(d) 長期再構成は自己回帰ではなく通常のfeedforward
  encoder-decoder+非自己回帰TCN融合（前回241で実装した「式(13)通りの自己回帰TCN」とは
  異なり、exposure biasの罠が無い）。
- 同じ研究室の後続研究(arXiv:2506.19358)は、自己教師あり事前学習(pretext task)による
  少データ対策(RFcardi)も提案しており、91トライアル全体を使った事前学習→fine-tuningという
  枠組みも今後の候補になる。
- なお、LifWavNet(IIT Kharagpur、Chen et al.・radarODEいずれとも無関係の第三の研究室)は
  MMECGではなく独自の臨床データセット(CR-RVS/Med-Radar)を使っており、直接の再現対象には
  ならないが、多重解像度分解という設計思想はSSTと共通する。
- **次の最優先タスクを更新**: データ入手待ちにせず、(1) SST(または近似としてCWT)による
  周波数領域への変換、(2) 50チャネル多数決PPI推定による1心拍単位へのタスク再定式化、
  (3) ODE型またはそれに準ずる振幅事前分布の実装、の3点にすぐ着手できる。

## radarODEの実装・再現結果: 目標に届かず、原因はシステム制約下の簡略化（2026-07-28）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「radarODEの実装・再現結果」節に記載。要点のみここに残す。

- 上記3点(SST変換、PPI多数決推定、ODE型振幅事前分布)を実装した2段パイプライン
  （SCEG=単一拍生成→長期再構成=非自己回帰TCN融合）を学習・評価した。
- **結果は目標(radarODE論文のPCC 87.9〜92.6%)に遠く届かず**、SCEG単一拍相関0.147・
  長期再構成窓内相関0.163に留まった（201のfeedforward CNN baseline 0.224にも届かず）。
- **最有力の原因はシステムRAM(15GB)の制約。** train+val(47トライアル、約1.86万拍)の
  SSTセグメントを当初の時間分解能(64、論文Table I相当)で全量メモリに載せたところ
  メモリ枯渇寸前になり(実際に一度学習を強制終了)、`T_FIXED_SST=16`まで縮小せざるを
  得なかった。1心拍をわずか16サンプルで表現することになり、Q/S波タイミング等の
  細部が失われたと考えられる。加えてtorchvision未導入によりDeformable Conv2dを
  通常Conv2dで代用、PPI推定もbiopeaksの簡易代替、という簡略化が重なった。
- **レシピ自体（SST・単一拍再構成・ODE事前分布・非自己回帰融合）の妥当性は否定されない。**
  独立した論文が同一データで実証済みという評価は変わらず、今回の未達は実装側の
  リソース制約に起因すると解釈するのが妥当。
- 次の一手（未実施）: Dataset を遅延ロード方式に変更し`T_FIXED_SST`を戻す、
  `torchvision`導入でDeformable Conv2dを使う、PPI推定をNeuroKit2の`ecg_findpeaks`系に
  差し替える。ただしユーザ指示により、次はECG基盤モデル/事前学習モデルによる
  PQRST形状事前分布の調査に移る。

## ECG基盤モデル/事前学習モデルによる形状事前分布の調査・実装（2026-07-28）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「ECG基盤モデル/事前学習モデルによる形状事前分布の調査・実装」節に記載。要点のみここに残す。

- 既存のECG基盤モデル(ECG-FM、SSSD-ECG等)を調査したが、分類向け(生成に不向き)・
  統合コストが高い等の理由で本セッションでの採用を見送り、**本リポジトリに既にある
  Schellenberger(ヒト30名、レーダ非依存の臨床ECG)で単一拍ECGオートエンコーダを自前で
  事前学習する**方針を採った(`data/ecg_beat_dataset.py`, `models/deep/ecg_beat_autoencoder.py`,
  `scripts/pretrain_ecg_beat_autoencoder.py`。24,809拍、val_loss 0.20まで収束)。
- `models/deep/radarode_sceg.py`に`shape_prior: "ode"|"pretrained"`オプションを追加し、
  McSharry型ODEデコーダを、事前学習済み(凍結)デコーダに差し替えた
  `configs/experiments/244_mmecg_radarode_sceg_pretrained_shape.yaml`で対照実験。
- **結果: 242(ODE, 0.147)と244(事前学習済み形状, 0.146)でほぼ同じ**。形状事前分布の
  種類を変えても改善しなかった。
- **結論: ボトルネックは形状デコーダの選択ではなく、backbone/encoderの時間分解能不足
  (T_FIXED_SST=16)にある可能性が高いことが、形状デコーダ側の要因を消去することで
  裏付けられた。** 事前学習済み形状デコーダのアイデア自体は無駄ではなく、backbone側の
  改善（遅延ロードDatasetでのT_FIXED_SST復元、torchvision導入でのDeformable Conv2d）を
  行った後に再度試す価値がある。

## SST時間分解能の復元とDeformable Conv2d導入（2026-07-28/29、config 245/246）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「SST時間分解能の復元(T_FIXED_SST 16→64)とDeformable Conv2d導入」節に記載。要点のみここに残す。

- 前節で特定した2つの簡略化点(`T_FIXED_SST=16`への縮小、Deformable Conv2dの通常Conv2d代用)
  を実際に解消した。`data/mmecg_singlecycle_dataset.py`を遅延ロード方式(trial単位LRU
  キャッシュ`_TrialSSTCache`+局所性を保った`TrialInterleavedSampler`)に全面書き換えし、
  メモリ制約(15GB RAM、実測ピークRSSは約900MBに削減)なしに`T_FIXED_SST=64`(原著と同じ値)へ
  復元。torchvisionを導入し`_DeformConv2dDownsampleBlock`でDeformable Conv2dを本来設計通り
  backboneへ実装した(`use_deformable: true`、既定)。
- **結果: SCEG単一拍相関 242(0.147)→245(0.209、+42%)。長期再構成窓内相関
  243(0.163)→246(0.268、+64%)。** 246の0.268は、feedforward CNN(201, 0.224)を含む
  本リポジトリ全体の波形回帰モデルの中で最高値になった。
- ただし目標(radarODE論文PCC 87.9〜92.6%)にはまだ遠い。245/246とも強い過学習
  (train_corrは0.7〜0.99まで到達するがval/testは0.2台で頭打ち)を示しており、
  モデル容量不足ではなく訓練プロセス・PPI推定精度側の課題が疑われる。
- 次の一手: PPI推定をNeuroKit2ベースへ置き換える(教師信号の質向上)、過学習対策
  (dropout・data augmentation)、複数seedでの再現性確認。

## タスク設計・前処理・学習設計の改善: アーキテクチャは支配的要因ではなかった（2026-07-29、config 247-249）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「タスク設計・前処理・学習設計の改善」節に記載。要点のみここに残す。

- **拍境界推定に開ループ位相ドリフトがあることを実測した。** 旧`_compute_bounds`は
  局所PPI"値"の分だけ前進を繰り返す方式で、trial 1で検証したところ真のR波位置との
  乖離が最大±0.46秒(ほぼ半心拍分)に達していた。`mmecg_ppi.py`に
  `detect_consensus_beat_times`(50chの候補ピーク時刻をガウシアンカーネルで積み上げた
  スパイク密度関数の極大点を拍タイミングとして直接採用)を追加し解消した。
- **アーキテクチャを一切変えず201と同じ凡庸なCNNにタスク設計の改善(単一拍再定式化+
  SST変換+新拍境界推定)だけを適用した(247, `models/deep/singlecycle_cnn.py`)ところ、
  単一拍相関0.218を達成し、凝ったアーキテクチャ(245のSCEG、Deformable Conv2d+ODE、
  旧タスク設計)の0.209とほぼ同水準になった。** アーキテクチャの工夫は副次的な要因に
  過ぎないことを示す直接的な証拠。
- `RadarODESCEG`に`dropout`オプションを追加し、新拍境界推定+dropout=0.2を組み合わせた
  248は単一拍相関0.258(245から+23%)を達成。長期再構成に組み込んだ249は窓内相関0.265
  (246の0.268とほぼ横ばい)で、**単一拍レベルの改善が窓レベルには比例して伝播しなかった**
  (長期再構成側の過学習・評価指標の性質等が原因と推測、詳細は上記報告書参照)。
- **結論: 目標(0.879〜0.926)との差は依然大きいが、「アーキテクチャをどれだけ精緻化しても
  届かない」という前提は再検討が必要。** 残るギャップの説明としては、拍境界推定の精度上限
  (真のR波とのwithin-100ms一致率が低いトライアルで数%程度)・実効訓練被験者数の少なさ・
  未実装の信号処理段(4Dビームフォーミング等)の方が有力。次はLeave-Subjects-Out CVへの
  切り替えと、長期再構成側への正則化・複数seed検証を優先する。

## 原著本文の再精読(4秒文脈窓・ECG正規化)は悪化という負の結果に終わった（2026-07-29、config 250/251）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「原著本文の再精読で見つけた2つの差分」節に記載。要点のみここに残す。

- radarODE論文本文(p.6)を精読し、"the actual SST segment is centered at the current cardiac
  cycle and expands to 4 seconds"という記述を発見。原著は出力(単一拍200サンプル)こそ
  タイトだが、**入力は拍を中心とした4秒間の文脈窓を丸ごと与えている**ことが分かった
  (我々の実装は入力もタイトな拍境界にクロップしていた)。
- 併せて、ECGのper-trial z-score正規化がトライアル間で振幅スケールを不整合にしていた
  問題(標準偏差が0.043〜0.293と最大6.9倍ばらつく)も発見し、固定スケール(0.15)正規化に
  修正した。train/eval間の処理不整合(リーク)は確認されなかった。
- **両方を実装し(`T_FIXED_SST`64→128、`CONTEXT_SEC=4.0`秒)、247/248の対照実験として
  250/251を学習したところ、いずれも悪化した**(247: 0.218→250: 0.195、
  248: 0.258→251: 0.220)。学習曲線はより極端な過学習を示した。
- **結論: 原著の記述を字句通り実装しても本実装・データ規模では再現しなかった。** 原因は
  未検証だが、epoch数・学習率が文脈拡大に伴うタスクの複雑化に追いついていない、
  backbone段数を4秒入力向けに再設計していない等が候補。**現時点の最良構成は引き続き248
  (単一拍相関0.258)。** 固定スケール正規化単体の効果は文脈窓化と混同されており未検証のまま
  残っている。

## RCGバンドパスフィルタ+正規化スコープの切り分け検証も悪化（2026-07-29/30、config 253/254）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「RCGバンドパスフィルタ+正規化スコープの切り分け検証も悪化」節に記載。要点のみここに残す。

- ユーザ指摘「振幅ドリフトはバンドパスフィルタ未適用が原因では」を検証したところ図星だった。
  RCG正規化(z-score)の分散計算が体動由来と推測される高周波ノイズ(25Hz超)に支配されており、
  10秒窓ごとのstdが生広帯域で最大9.3倍ばらつくのに対し、[1,25]Hzバンドパス後は3.4倍まで
  縮小することを実測した。
- `data/bandpass.py`(ゼロ位相バタワース)を実装し、`mmecg_sst_cache.py`のRCG正規化前に
  適用するオプションを追加(別キャッシュディレクトリ`mmecg_sst_bp/`)。また正規化統計量の
  計算範囲を選べる`norm_scope`("trial"|"beat")を追加した(単一拍の生信号から都度SST計算
  すると6時間規模で非現実的なため、"beat"はSST計算自体はtrial全体で行い事後に再正規化する
  妥協実装)。ECGへのバンドパスはQRSの鋭さを潰すリスクがあるため見送った。
- **248(基準、0.258)に対し、253(バンドパス+trial-scope)は0.240(-7%)、254(バンドパス+
  beat-scope)は0.217(-16%)といずれも悪化した。** これで4秒文脈窓に続き、信号解析・
  paper読解に基づく改善仮説が2件連続で悪化するという結果になった。
- **結論: 現在のパイプライン(248: タイトクロップ+per-trial z-score+dropout+consensus PPI)
  が既にかなり良いローカルオプティマムに近く、個別の前処理仮説の単発追加では超えられない
  可能性が高い。** 次はLeave-Subjects-Out CV・複数seed再現性確認・訓練データ規模拡充等の
  より構造的な改善を優先すべき。

## タイミングアライメント補正の一連の試み: すべて失敗、しかし伸びしろの実測に成功（2026-07-30、config 255・256・258・259）

詳細は[`reports/mmecg_comparison/prior_work_accuracy_comparison.md`](reports/mmecg_comparison/prior_work_accuracy_comparison.md)
「タイミングアライメント補正の一連の試み」節に記載。要点のみここに残す。

- **診断**: 248のtest予測に±20〜30サンプルのシフト探索を行うと相関0.36→0.54まで跳ね上がる
  ことを確認(シフト方向は左右対称)。radarODE論文のPCC計算にDTW/ラグ探索は見つからず
  (評価式は同じと推測)、ただし原著のODEデコーダは学習可能な時間遅延τで生成波形をシフトする
  設計を持っており、我々の実装はこれを省略していた。
- **相関損失(255)**: MSEを`1-corr`に置き換えたが248(0.258)を下回る0.239。勾配が弱く
  収束が遅い。
- **学習可能なτ(256/258、Temporal Transformer layer着想)**: v1(グローバルpooling)は
  test_corr 0.278(対照257の0.203から+37%)と一見改善したが、**学習されたτは平均-0.45
  ±0.09サンプルとほぼ一定値で、サンプルごとの意味ある補正を全く学習していなかった。**
  位置情報を保持したv2は0.199とむしろ悪化。
- **オラクル上限の実測**: 257の予測に正解を使った全探索シフトを適用すると相関
  0.203→**0.586**(+189%)。oracle shiftの分布(絶対値平均13.0サンプル)は256/258の
  ほぼ一定値(0.4〜0.5サンプル)と全く異なり、256の改善が偶然だったことを裏付ける。
  **同時に「正確な個別アライメントができれば単一拍相関は0.58程度まで伸びる」という
  重要な上限を確認できた。**
- **Shift-invariant training(259、学習時のみoracleでアライメントしてから損失計算)**:
  test_corr 0.084という最悪の結果。**モデルが生出力を正しい位相に置く動機を完全に失う**
  という新たな失敗モードを発見(train_lossは下がるが評価は生出力に対して行うため崩壊)。
- **結論**: 3つのアプローチはいずれも実際には機能しなかったが、0.258→0.586という大きな
  伸びしろを実測できた。次善の設計案(優先順)は(1)シフト幅を制限したshift-invariant
  training、(2)学習後半でシフト幅を0へ減衰させるカリキュラム、(3)疑似ラベルτへの
  直接教師あり補助損失、(4)出力ではなく入力側(SSTクロップ)を補正する設計。

### 追試(2026-07-30、config 260・261): シフト幅制限は成功、τ教師あり補助損失は「τ精度」では説明できない改善

上記(1)と(3)を追試(詳細は報告書「追試: シフト幅制限とτの教師あり補助損失」節)。

- **260(shift幅を±30→±5に制限)**: test_corr **0.231**で257(0.203)を上回り、
  無制約(259, 0.084)の破滅的失敗と対照的に成功。優先順位1位の提案が有効と確認。
- **261(τをoracle疑似ラベルへ直接MSE回帰する補助損失を追加、weight=1.0)**: test_corr
  **0.245**でさらに257を上回ったが、学習後のτ予測は**全例で完全に同一の定数(std=0.0000)**、
  oracleτとの相関も**-0.0001**(ほぼ無相関)。つまり改善は「τを正確に予測できた」から
  ではなく、256と同様の「ほぼ定数の小さな一律シフトへの収束」による。
- **262(τ補助損失の重みを50倍に引き上げ、weight=50.0)**: 261の重みでは再構成損失に対し
  τ損失のスケールが2桁小さく勾配が実効的に無視されていた可能性を検証したが、test_corr
  **0.224**で261より悪化。τ予測のstdは0.0000→0.0035とわずかに非ゼロになったが、
  oracleτとの相関は**-0.0208**でやはりほぼ無相関(ノイズが増えただけ)。重みを上げる
  という単純な対策では「正しい」τ予測には近づかなかった。
- **結論**: ユーザが提示した「まず正確にτを予測し0.586に近づける」目標は、weight=1.0/
  50.0のいずれの教師あり補助損失実装でも未達成。260(シフト幅制限)が今のところ唯一、
  257を上回りかつ機序が理解できているアプローチ。

### τ専用アーキテクチャ(TauOnlyCNN)による純粋なτ回帰診断(2026-07-31、config 263)

「τ疑似ラベルとの相関がゼロなら波形精度が上がらないのは当然」というユーザ指摘を受け、
波形は一切予測せずτの疑似ラベルのみを回帰する専用モデル`TauOnlyCNN`
(SST(50,F,T)を周波数軸を潰さず2D Conv2dで処理、family=`mmecg_tau_predictor`)を新設。
261/262の失敗が「補助損失の重み付け」の問題か「波形回帰用アーキテクチャがτ推定に
向いていない」のかを切り分けた。

- **train_corr**: 0.15〜0.30で推移(261/262のtrain相関ほぼ0より遥かに強い信号)。
  → アーキテクチャの表現力不足ではなく、train内では学習できることを確認。
- **val_corr(best)=0.073、test_corr=0.061**: train/valの間に大きなギャップ、
  汎化性能が課題。
- **結論**: アーキテクチャを変えるとtrain相関は劇的に改善するが、val/testへの
  汎化はまだ弱い(0.586目標には遠い)。原因候補は疑似ラベル自体のノイズ、
  訓練被験者数(7頭)の少なさによる過学習、SST入力の汎化可能なタイミング情報の
  不足。次善策は疑似ラベルの質の見直し、正則化強化、LOSO評価。
- **追記(相関ではなくMAEで再評価)**: ユーザ指摘「スカラー回帰なら相関より誤差で
  評価すべき」を受け%表示のMAEを算出したところ、**モデルのtest MAE=6.63%は
  τ=0(補正なし)を予測するベースラインのMAE=6.52%より悪化**していた
  (test_corr=0.061という弱い正の相関は実用的な予測精度の向上を意味していなかった)。
  相関だけでスカラー回帰の成否を判断してはならないという教訓。
- **論文本文の確認**: radarODE論文のτは、正解ECGを使ったオフライン補正ではなく
  **レーダー由来の潜在特徴から線形層で予測される、完全にエンドツーエンドで学習可能な
  パラメータ**(ODE形状パラメータηと同じ潜在特徴から同時に予測)であり、推論時に
  正解ECGを使わないオンライン推論可能な設計。これは256/258(学習可能なτヘッド)と
  同系統だが、ODEの5ピーク・パラメトリックな形状制約により「形状で誤魔化す」余地が
  少なく、τに正しい勾配が伝わりやすい可能性がある(詳細は報告書参照)。

### 原著設計の再現(SCEG本体へのτ組み込み、2026-07-31、config 264/265)

ユーザ指摘「疑似ラベルτが波形予測モデル257の癖に依存してしまっている」を受け、
「最重要事項は論文が実際にやっているτの予測方法を再現すること」という優先順位のもと、
`radarode_sceg.py`の`ODEDecoder`にτを追加した(原著再現、疑似ラベル一切不使用、
最終再構成損失のみで学習)。判明した事実: **現在の実装にはτが完全に欠落していた**
(位相グリッドを固定linspaceで与えていた)。

248相当(80epoch)を高速反復用に25epochへ短縮した264(τあり)とcontrol 265(τなし)を
比較:

| config | val_corr(best) | test_corr | τの分布 |
|---|---|---|---|
| 264(τあり) | **0.204** | 0.232 | mean=0.137, std=0.019, max=0.150(上限に張り付き) |
| 265(τなし) | 0.157 | **0.253** | – |

valでは264が優位だがtestでは265が僅かに優位という一貫しない結果。τは
`tau_max_frac=0.15`の上限にほぼ張り付いた、個別サンプルに依存しないほぼ定数の
バイアスに収束しており、256/258と質的に同じ失敗パターン。**原著の設計を忠実に
再現しても、τが「個別のタイミング補正」に収束しない問題は解消されなかった。**
248本体(80epoch)での検証は優先度を下げる。

### 疑似ラベル手法の比較 + 公式実装調査による方針転換(2026-07-31、config 266)

ユーザ指示「疑似ラベル手法を比較してからτ collapseの根本原因を見直す。mmWaveが情報を
持っていない可能性もあるが原著を読む限りそうではないはず、関連研究にヒントがあるかも」を
受け、3方向で調査した。

**(A) model-independent疑似ラベルとの比較**: 257(特定の波形回帰モデル)の予測に依存しない
新ラベル`rpeak_position`(拍境界内でのR波の実際の位置、`tau_pseudo_labels.py`)を実装し、
263(oracle_shift、257基準)と同条件で対照実験(266)。

| ラベル方式 | config | モデルMAE(test) | ゼロ予測MAE | 訓練平均定数予測MAE | corr(test) |
|---|---|---|---|---|---|
| oracle_shift(257基準) | 263 | 6.63% | 6.52% | – | 0.061(MAEでは負け) |
| rpeak_position(独立) | 266 | 11.27% | 16.52% | **9.99%** | 0.048 |

266はゼロ予測には勝ったが(=集団全体に系統的なτバイアスは実在する)、「訓練データの平均値を
一定予測」というベースラインには負けた。**疑似ラベルの生成方法を根本的に変えても、
「個別サンプルの情報を活用できず、集団平均への収束に留まる」という同じ失敗パターンが
再現した。** ラベルの質(257依存かどうか)が主因ではないことを示す。

**(B) 関連研究**: Spatial Transformer Networksのlocalization networkは既知の学習困難性
(identity変換近傍への収束)を持つ。radarODEの同著者グループによる後継論文
radarODE-MTL(arXiv:2410.08656)は、「ECG波形再構成という難しいタスクは他のタスクより
勾配が弱く学習が遅れる」という課題を明示的に報告し、専用の最適化手法(Eccentric Gradient
Alignment)まで提案している——著者ら自身がタスク間の勾配競合・難易度不均衡を認めている。

**(C) 決定的な発見**: radarODEの公式実装(GitHub `ZYY0844/radarODE-MTL`)のコードを直接
確認したところ、`ECGParameterEstimator`が出力するのはP/Q/R/S/T各波の`(a,b,θ)`×5=15
パラメータのみで、**独立した「τ」というスカラー変数はコード上どこにも存在しなかった**
(`shift`/`delay`/`align`で全文検索しても該当なし)。論文本文の「ODEの解がτだけ左に
シフトする」という記述は、5つの`θ_i`(各波の位相中心)を動かした結果を説明した言い回しに
過ぎず、専用ヘッドの実装は無いと考えられる。**つまり256〜266で行ってきた「τ専用ヘッドを
追加して原著を再現する」試み自体が、論文文章の読み方に基づく誤った拡張だった可能性が高い。**
248(τ専用ヘッド無し、15パラメータのみ)の方が公式実装に忠実であり、追加した独立τヘッドは
既存の`θ_R`等と機能的に重複する縮退した自由度を持ち込んだだけと考えられる。

**結論**: τ専用ヘッドの追加は深追いせず、248相当(15パラメータのみ)を基本線に戻す。
拍ごとの機械的-電気的タイミング差は、現在のbackbone/encoderの時間分解能・SNRでは
(集団全体の平均バイアスを除けば)個別に推定するための情報が十分取り出せていない
可能性が高い。詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照。

### τ・Anchor補助タスクの公式実装への忠実化(2026-07-31、config 267/268)

ユーザ指示「τだけ公式実装に倣って実装し精度変化を見る、次に波形回帰タスク全体も忠実に
再現したケースを作る」を受け実施。フル再現(4秒複数拍窓+DCNResNet+PPI/Anchor+LibMTL/EGA)
は数時間〜数日規模のため、ユーザに再現範囲を確認し「既存の1心拍パイプラインにAnchor
補助タスクだけ追加」という中間案を採用。

**267(τの公式実装忠実化)**: `OfficialODEDecoder`を新設し、公式`ECGParameterEstimator`の
深い全結合スタック・`default_input`・`scale_output`(±100%乗算スケーリング)を移植、τ専用
ヘッドは廃止。265(独自パラメータ化・τなし)との比較:

| config | val_corr(best) | test_corr |
|---|---|---|
| 265(独自) | 0.157 | **0.253** |
| 267(公式) | **0.192** | 0.210 |

valとtestで優劣が入れ替わり、264/265と同じ不安定さが再現。公式に忠実化しても大勢に
影響する改善は無し。

**268(Anchor補助タスク追加)**: `AnchorHead`を新設し、`tau_pseudo_labels.compute_rpeak_heatmap_labels`
(拍境界内のR波位置にガウシアンheatmapを立てる、model-independent)を教師信号にMSE補助損失
(anchor_weight=1.0)を追加。265との比較:

| config | val_corr(best) | test_corr | anchor_loss(train) |
|---|---|---|---|
| 265(Anchorなし) | 0.157 | 0.253 | – |
| 268(Anchor追加) | 0.175 | **0.254** | 0.053→0.033(順調に減少) |

anchor_lossは順調に減少(R波位置自体はある程度学習できている)が、主タスクのtest_corrは
265とほぼ同じで改善は誤差の範囲内。**τヘッド・公式パラメータ化・Anchor補助タスクのいずれの
形でタイミング情報をモデルに与えても、波形再構成の主タスクには系統的な改善が見られなかった。**
これは現在のbackbone/encoderが1心拍タイトクロップ窓から抽出できる特徴量自体に、拍ごとの
タイミングを個別に区別する情報が十分残っていないという結論を補強する。詳細は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照。

### 時間分解能不足の解消: backbone downsampling段数削減(2026-08-01、config 269/270)

ユーザ指示「ではその問題を解決してください」を受け対応。実測(trial 1、203拍)の結果、
タイトクロップの生SST長は平均87・中央値89フレームで、`T_FIXED_SST=64`は生の時間分解能を
ほぼ保っていた。問題は後段の`Backbone`が3段downsampling(各stride2、64→32→16→8)で
時間軸を潰しすぎていたこと(encoder出力が1拍あたり8フレーム=約87ms/フレームまで潰れる)。

対策として、`T_FIXED_SST`を上げる案(生クロップが平均87フレームしか無いため、128への
リサイズは見かけ上の解像度に過ぎず不採用)ではなく、**`backbone_channels`を3段から2段
(`[128,256,512]`→`[128,256]`)に減らし、downsampling段数を1段減らす**(64→32→16、
`encoder_out_time`8→16)方式を採用した。これは補間ではなく実際の情報保持量を増やす変更。

268(3段、Anchor有、25epoch)との比較(269、2段、他同一):

| config | encoder_out_time | val_corr(best) | test_corr | anchor_loss(train) |
|---|---|---|---|---|
| 268(3段) | 8 | 0.175 | 0.254 | 0.053→0.033 |
| 269(2段) | 16 | **0.240** | **0.300** | 0.053→**0.020** |

これまでの一連の実験(264〜268)がすべてval/testで優劣が入れ替わる不安定な結果だったのに
対し、269は**valとtestの両方が一貫して268を上回った**。248(80epoch、test_corr=0.258)も
25epochの短縮検証で上回った。80epochのフル学習(270、269と同一構成)でも改善は維持:
val_corrはepoch29で最良0.235に達し以降過学習のみ進行(test_corr=0.250)。25epoch短縮検証
(269)とほぼ同じ水準で、epoch数を伸ばしても消えない一貫した効果と確認できた。

**「backbone/encoderの時間分解能不足」という仮説が、実測による裏付けと、downsampling
段数削減という単純な変更による一貫した改善の両面から支持された。** τ専用ヘッドや公式
パラメータ化、Anchor補助タスク単体では改善しなかった問題が、入力側の時間分解能を上げる
だけで解消した。詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照。

### 4秒複数拍窓の再検証・時間分解能のさらなる引き上げ(2026-08-01、config 271/272)

「4秒窓の利点は?」というユーザ質問に、(a)拍境界推定ノイズへの依存低減、(b)Anchorタスクの
ダイナミックレンジ拡大、の2点を回答。250/251では旧アーキテクチャで悪化していたが、269の
改善後構成で再検証。合わせてBackboneに非対称stride機能(周波数軸・時間軸を別々に指定可能)
を追加し、時間分解能をさらに倍増する単体構成(272)も検証。

**注意**: 271/272を最初並列実行したところ、`make_run_dir`が秒単位timestampのみでrun_idを
作る仕様のためrun_dir衝突が発生し、結果が汚染された。破棄して逐次実行し直した。

| config | 窓 | encoder_out_time | val_corr(best) | test_corr |
|---|---|---|---|---|
| 269 | タイトクロップ | 16 | **0.240** | **0.300** |
| 271(4秒窓) | 4秒複数拍窓 | 32 | 0.143 | 0.208 |
| 272(時間分解能倍増) | タイトクロップ | 32(非対称stride) | 0.229 | 0.265 |

**271(4秒窓)は269より明確に悪化**(anchor_lossも0.041と269の2倍で高止まり)。250/251の
結果が改善後構成でも再現し、この課題ではタイトクロップの方が一貫して優れると確認。
**272(時間分解能をさらに倍増)も269よりわずかに悪化**——時間分解能の改善は「無いより
ある方が良い」が「多いほど良い」わけではなく、現状の構成では16フレーム前後が妥当な
落とし所。269が引き続き最良構成。詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照。

### まとめ: 269を採用構成として確定(2026-08-02)

271・272がいずれも269を上回らなかったため、**`configs/experiments/269_mmecg_radarode_sceg_hires.yaml`
を現時点の採用構成として確定**(以後の実験のベースライン)。271・272の設定ファイルは
負の結果の記録として残すが不採用。

**今回の調査で最も重要な手がかりになったのは、radarODE/radarODE-MTL論文の著者ら自身が
公開している公式実装コード**([GitHub: ZYY0844/radarODE-MTL](https://github.com/ZYY0844/radarODE-MTL)、
対応論文: radarODE [arXiv:2408.01672](https://arxiv.org/abs/2408.01672)、radarODE-MTL
[arXiv:2410.08656](https://arxiv.org/abs/2410.08656))。論文本文の記述だけでは「τがなぜ
定数に収束するのか」の原因を特定できなかったが、公式コード(`ODE_solver.py`・`decoder.py`・
`spectrum_dataset.py`・`utils/utils.py`)を直接読んだことで、τ専用パラメータが実は存在
しないこと、"Anchor"補助タスクの実在、4秒複数拍窓という入力設計など、論文の文章からは
読み取れなかった実装詳細が判明し、調査の方向性(267/268)、そして最終的な原因特定(269の
時間分解能改善)につながった。経緯の詳細な要約表は
`reports/mmecg_comparison/prior_work_accuracy_comparison.md`「まとめ」節参照。
