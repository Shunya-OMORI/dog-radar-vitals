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
