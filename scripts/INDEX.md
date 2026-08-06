# scripts/ INDEX

dog-radar-vitals/scripts/ の実行スクリプトを責務ごとに整理した索引。
本体コードは`src/dog_radar_vitals/`側、ここは学習外の解析・比較・ベンチマーク用CLI。

## 効率計測(エッジデバイス、RPi5展開時に必読)

| script | 概要 |
|---|---|
| `benchmark_edge_efficiency_gnn.py` | エッジ実行性の実測(空間GNN版)。 |
| `benchmark_model_efficiency.py` | ecg_registryの各モデルについて、パラメータ数・推定FLOPs・CPU推論レイテンシを測定する。 |

## モデル横断比較・レポート生成

| script | 概要 |
|---|---|
| `compare_ecg_vs_rpeak.py` | 波形回帰(101 ecg_cnn1d)とheatmap回帰(102 rpeak_cnn1d)を、単発の学習run同士で比較する。 |
| `compare_mmecg_models.py` | MMECG(mmWave)向けの全モデル系統を横断比較する。 |
| `make_comparison_report.py` | run_comparison.pyが生成したマニフェストを読み、比較表とグラフを reports/ に出力する。 |
| `plot_efficiency_pareto.py` | benchmark_model_efficiency.pyとcompare_mmecg_models.pyの出力を突き合わせ、 |
| `plot_efficiency_pareto_ecg_only.py` | 波形推定モデル(ecg_*系)だけに絞った「精度(波形相関) vs 効率性(パラメータ数)」Pareto図。 |
| `plot_efficiency_pareto_ecg_only_rrmae.py` | 波形推定モデル(ecg_*系)だけに絞った「RR Interval MAE vs 効率性」Pareto図。 |
| `plot_rr_mae_ecg_only.py` | 波形推定モデル(ecg_*系)だけに絞ったRR Interval MAEランキング。 |
| `plot_waveform_comparison.py` | 波形推定モデル(ecg_*系、mmecg_seq2seq/mmecg_spatial_seq2seq family)だけを集めて、 |
| `plot_learning_curves.py` | 複数runのval_mae学習曲線を重ねて描く。 |
| `plot_ncp_differentiation.py` | ノイズ頑健性実験・データ効率実験の結果を可視化する（NCPを別軸で評価する狙い、 |
| `plot_heatmap_teacher_vs_inference.py` | heatmapキーポイント検出(102_rpeak_cnn1d)の「教師信号」と「推論時の極大点検出」の対応を図示する。 |

## ベースライン・評価プロトコル

| script | 概要 |
|---|---|
| `compute_baselines.py` | 固定のtrain/val/test犬分割（configs/base.yaml）に対するtrivial/oracleベースラインを計算する。 |
| `evaluate_arousal_from_hr_case.py` | CASEデータセットで「HR由来の覚醒度スコア」が連続arousalアノテーションと相関するかを検証する。 |
| `evaluate_noise_robustness.py` | 学習済みの波形回帰モデル(ecg_ncp / ecg_cnn1d / ecg_unet1d等)に、テスト時にRCG入力へ |
| `evaluate_per_beat_correlation.py` | Chen et al. (2022, RCG2ECG)の相関評価方法を再現した波形相関の再評価。 |
| `diagnose_within_dog_signal.py` | あるrunが「個体内の時間変動」を実際に追えているかを診断する。 |
| `test_cv_significance.py` | cross-validationのmanifestを読み、モデルがtrivialベースラインに対して統計的に有意な差を |

## データ前処理・事前計算

| script | 概要 |
|---|---|
| `precompute_mmecg_sst.py` | 全91トライアルのSST(50ch)を事前計算し`data/processed/mmecg_sst/`にキャッシュする。 |
| `pretrain_ecg_beat_autoencoder.py` | 単一拍ECGオートエンコーダをSchellenberger(ヒト30名、レーダ非依存の臨床ECG)で事前学習する。 |

## 学習・CV実行オーケストレーション

| script | 概要 |
|---|---|
| `run_comparison.py` | 複数configを一括学習・評価し、run_dir一覧をマニフェストとして保存する。 |
| `run_dog_cross_validation.py` | 犬10頭をn_folds個のfoldに分け、犬を入れ替えながら学習・評価するleave-few-dogs-out CV。 |
| `run_ecg_cross_validation.py` | 101(波形回帰)と102(heatmap回帰)を、被験者を入れ替えたn-fold cross-validationで比較する。 |
| `run_radarode_sceg_lodo_cv.py` | config269(採用構成)を11名全員でLeave-One-Subject-Out CVにかけ、test_corrのばらつきを見る。 |
