# configs/experiments/ INDEX

99個の実験configを責務帯（番号帯）ごとに整理した索引。
各configの先頭コメントから抽出。詳細な経緯・比較結果は`reports/`配下の各報告書、
特にR波heatmap系(273-281)は`findings_2026-08-07_rpeak_heatmap_investigation.md`を参照。

運用ルール: 新規configは既存の番号帯に追記し、先頭コメントに「何と比較し何を変えたか」を
必ず書く（CLAUDE.md R4の単一変数原則の記録として）。番号帯を跨ぐ新テーマは次の空き番号帯から始める。

## 001-016: 犬HR/BR予測ベースライン（本題、Transformer/CNN1D/LSTM/古典ML）

| config | 概要 | 判定/状態 |
|---|---|---|
| `001_transformer_hr` | 001: 素朴なTransformer Encoderによる心拍数(HR)予測のベースライン |  |
| `002_transformer_br` | 002: 素朴なTransformer Encoderによる呼吸数(BR)予測のベースライン |  |
| `003_cnn1d_hr` | 003: 1次元CNNによる心拍数(HR)予測 |  |
| `004_cnn1d_br` | 004: 1次元CNNによる呼吸数(BR)予測 |  |
| `005_lstm_hr` | 005: 双方向LSTMによる心拍数(HR)予測 |  |
| `006_lstm_br` | 006: 双方向LSTMによる呼吸数(BR)予測 |  |
| `007_ridge_hr` | 007: Ridge回帰による心拍数(HR)予測 |  |
| `008_ridge_br` | 008: Ridge回帰による呼吸数(BR)予測 |  |
| `009_random_forest_hr` | 009: Random Forestによる心拍数(HR)予測 |  |
| `010_random_forest_br` | 010: Random Forestによる呼吸数(BR)予測 |  |
| `011_gradient_boosting_hr` | 011: Gradient Boostingによる心拍数(HR)予測 |  |
| `012_gradient_boosting_br` | 012: Gradient Boostingによる呼吸数(BR)予測 |  |
| `013_transformer_hr_lr5e-4` | 013: Transformer(HR)の学習率対照実験その1 |  |
| `014_transformer_hr_lr1e-3` | 014: Transformer(HR)の学習率対照実験その2 |  |
| `015_transformer_hr_epochs300` | 015: Transformer(HR)のエポック数対照実験 |  |
| `016_transformer_hr_warmup_cosine` | 016: Transformer(HR)のスケジューラ対照実験 |  |

## 101-102: ヒトデータ(Schellenberger)での手法検証（波形回帰 vs heatmap定式化）

| config | 概要 | 判定/状態 |
|---|---|---|
| `101_ecg_cnn1d_resting` | 101: レーダI/Q -> ECG波形推定（ヒト、Schellenberger et al. 2020データセット、Restingシナリオ） |  |
| `102_rpeak_cnn1d_resting` | 102: レーダI/Q -> R波heatmap（疎なイベント検出） |  |

## 201-213: MMECG(mmWave) 波形回帰/heatmapのモデル横断比較（CNN/Transformer/LSTM/NCP/複素CNN/古典ML）

| config | 概要 | 判定/状態 |
|---|---|---|
| `201_mmecg_cnn1d_waveform` | 201: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `202_mmecg_rpeak_cnn1d` | 202: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `203_mmecg_transformer_waveform` | 203: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `204_mmecg_rpeak_transformer` | 204: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `205_mmecg_lstm_waveform` | 205: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `206_mmecg_rpeak_lstm` | 206: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `207_mmecg_ncp_waveform` | 207: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `208_mmecg_rpeak_ncp` | 208: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `209_mmecg_complex_cnn_waveform` | 209: RCG(mmWave, 50ch)のanalytic signal(Hilbert変換)-> ECG波形推定 |  |
| `210_mmecg_rpeak_complex_cnn` | 210: RCG(mmWave, 50ch)のanalytic signal(Hilbert変換)-> R波heatmap推定 |  |
| `211_mmecg_classical_ridge_rr` | 211: RCG(mmWave, 50ch) -> 窓内平均RR Interval[ms] |  |
| `212_mmecg_classical_random_forest_rr` | 212: RCG(mmWave, 50ch) -> 窓内平均RR Interval[ms] |  |
| `213_mmecg_classical_gradient_boosting_rr` | 213: RCG(mmWave, 50ch) -> 窓内平均RR Interval[ms] |  |

## 214-239: PQRSTグラフGNN・空間融合(posXYZ)・GAN拡張・データ効率実験

| config | 概要 | 判定/状態 |
|---|---|---|
| `214_mmecg_beatgraph_gnn` | 214: RCG beat segment(50ch, R波中心±0.4秒) -> PQRST 5点グラフ(時刻オフセット, 振幅) |  |
| `215_mmecg_beatgraph_gnn_gan` | 215: 214と同一タスク・データだが、生成器(BeatGraphGNN)を判別器と敵対的に学習するGAN拡張 |  |
| `216_mmecg_complex_cnn_v2_waveform` | 216: RCG(mmWave, 50ch)のanalytic signal -> ECG波形推定 |  |
| `217_mmecg_rpeak_complex_cnn_v2` | 217: RCG(mmWave, 50ch)のanalytic signal -> R波heatmap推定 |  |
| `218_mmecg_conformer_waveform` | 218: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `219_mmecg_rpeak_conformer` | 219: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `220_mmecg_unet1d_waveform` | 220: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `221_mmecg_rpeak_unet1d` | 221: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `222_mmecg_heatmap_gan` | 222: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `223_mmecg_rpeak_ncp_v2_lr1e-4` | 223: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `224_mmecg_ncp_v2_ep50` | 224: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `225_mmecg_ncp_n2` | 225: ecg_ncpのデータ効率実験(train被験者2名版) |  |
| `226_mmecg_ncp_n4` | 226: ecg_ncpのデータ効率実験(train被験者4名版) |  |
| `227_mmecg_cnn1d_n2` | 227: ecg_cnn1dのデータ効率実験(train被験者2名版) |  |
| `228_mmecg_cnn1d_n4` | 228: ecg_cnn1dのデータ効率実験(train被験者4名版) |  |
| `229_mmecg_unet1d_n2` | 229: ecg_unet1dのデータ効率実験(train被験者2名版) |  |
| `230_mmecg_unet1d_n4` | 230: ecg_unet1dのデータ効率実験(train被験者4名版) |  |
| `231_mmecg_spatial_fusion_waveform` | 231: RCG(mmWave, 50ch) + posXYZ(50点の3D座標) -> ECG波形推定 |  |
| `232_mmecg_rpeak_spatial_fusion` | 232: RCG(mmWave, 50ch) + posXYZ -> R波heatmap推定 |  |
| `233_mmecg_conv_ncp_waveform` | 233: RCG(mmWave, 50ch) -> ECG波形推定 |  |
| `234_mmecg_rpeak_conv_ncp` | 234: RCG(mmWave, 50ch) -> R波heatmap推定 |  |
| `235_mmecg_spatial_gnn_waveform` | 235: RCG(mmWave, 50ch) + posXYZ -> ECG波形推定 |  |
| `236_mmecg_rpeak_spatial_gnn` | 236: RCG(mmWave, 50ch) + posXYZ -> R波heatmap推定 |  |
| `237_mmecg_spatial_heatmap_gan` | 237: RCG(mmWave, 50ch) + posXYZ -> R波heatmap推定 |  |

## 240-251: radarODE(Zhang et al.)SCEG/長期ECG再構成の再現、正規化方式・アーキ探索

| config | 概要 | 判定/状態 |
|---|---|---|
| `240_mmecg_cnn1d_minmax` | 240: 201(ecg_cnn1d, z-score正規化)の正規化方式のみをmin-max([-1,1])に変えた対照実験 |  |
| `241_mmecg_chen2022_repro` | 241: Chen et al. (2022, RCG2ECG、同一データセットの原著論文)アーキテクチャの忠実な再現 |  |
| `241_mmecg_cnn1d_strong_reg` | 241: 201/227/228/240系(ecg_cnn1d, MMECG連続波形回帰)は全試行でtrain corr 0.92-0.99 に対しval co |  |
| `242_mmecg_cnn1d_amplitude_weighted_loss` | 242: 241(正則化強化)はtrain/val corrギャップを実質的に縮められなかった(2026-08-05) |  |
| `242_mmecg_radarode_sceg` | 242: radarODE(Zhang et al. 2024/2025, arXiv:2408.01672)のSCEG(単一拍ECG再構成)を再現 |  |
| `243_mmecg_radarode_longterm` | 243: radarODEの長期ECG再構成ネットワーク(Table III・非自己回帰TCN融合) |  |
| `243_mmecg_rpeak_spatial_gnn_lowds` | 243: 236(空間GNN、本セッション最良RR-MAE=9.67ms、パラメータ34Kと極小)のF1が 0.383と低めな弱点の原因調査 |  |
| `244_mmecg_radarode_sceg_pretrained_shape` | 244: 242(SCEG)のshape_prior(McSharry型ODEデコーダ)を、ユーザ提案の 「ECG基盤モデル/事前学習モデル」路線に差し替えた対 |  |
| `245_mmecg_radarode_sceg_full_res` | 245: radarODE SCEG(単一拍ECG再構成)、T_FIXED_SST復元+Deformable Conv2d版 |  |
| `246_mmecg_radarode_longterm_full_res` | 246: radarODEの長期ECG再構成ネットワーク、245(T_FIXED_SST=64+Deformable Conv版SCEG)を使用 |  |
| `247_mmecg_singlecycle_cnn` | 247: 単一拍SST->ECG回帰を、201と全く同じ凡庸なCNN(ECGWaveformCNN1Dそのもの)で解く |  |
| `248_mmecg_radarode_sceg_ppi_dropout` | 248: 245(T_FIXED_SST=64+Deformable Conv2d)に、タスク設計の改善(拍境界のコンセンサス検出、 `mmecg_single |  |
| `249_mmecg_radarode_longterm_ppi_dropout` | 249: radarODEの長期ECG再構成ネットワーク、248(拍境界コンセンサス検出+dropout版SCEG)を使用 |  |
| `250_mmecg_singlecycle_cnn_context` | 250: 247(単一拍タスク+凡庸CNN)に、原著本文(radarODE p.6)で確認した「4秒文脈窓」入力と 固定スケールECG正規化を追加した版 |  |
| `251_mmecg_radarode_sceg_context` | 251: 248(拍境界コンセンサス検出+dropout版SCEG)に、4秒文脈窓入力+固定スケールECG正規化を 追加した版 |  |

## 253-266: SCEG改善サイクル（バンドパス、拍境界コンセンサス、dropout、τ補正、oracle shift）

| config | 概要 | 判定/状態 |
|---|---|---|
| `253_mmecg_radarode_sceg_trial_bandpass` | 253: 248(タイトクロップ+dropout+consensus PPI、単一拍相関0.258)に、RCG正規化前の [1,25]Hzバンドパスフィルタを追 |  |
| `254_mmecg_radarode_sceg_beat_bandpass` | 254: 253と同一だが、正規化スコープを拍セグメント自身(norm_scope="beat")に変更した版 |  |
| `255_mmecg_radarode_sceg_corrloss` | 255: 248(タイトクロップ+dropout+consensus PPI、単一拍相関0.258、MSE損失)のMSE損失を 微分可能なPearson相関損失 |  |
| `256_mmecg_singlecycle_cnn_tau` | 256: 247(タイトクロップ+凡庸CNN、単一拍相関0.218)に、学習可能な時間遅延τ補正 (TemporalShiftHead)を追加した高速反復用co |  |
| `257_mmecg_singlecycle_cnn_control25ep` | 257: 256と全く同じ設定(25epoch、凡庸CNN)だがuse_temporal_shift=falseの対照実験 |  |
| `258_mmecg_singlecycle_cnn_tau_v2` | 258: 256(TemporalShiftHead v1)の再設計版 |  |
| `259_mmecg_singlecycle_cnn_oracleshift_train` | 259: 257(対照、25epoch、シフト無し)に、shift-invariant training(学習時のみoracle shift を適用してから損失 |  |
| `260_mmecg_singlecycle_cnn_oracleshift_train_limited` | 260: 259(shift-invariant training, oracle_max_shift=30)は無制約すぎて生出力の位置合わせ 動機を失い、te |  |
| `261_mmecg_singlecycle_cnn_tau_supervised` | 261: τの教師あり補助損失(tau supervision) |  |
| `262_mmecg_singlecycle_cnn_tau_supervised_w50` | 262: 261(tau_supervision weight=1.0)はτが定数に収束し(std=0.0000)、oracleτとの相関が ほぼ0だった |  |
| `263_mmecg_tau_predictor_standalone` | 263: τのみを予測する専用モデル(TauOnlyCNN)を、τ疑似ラベルへの回帰損失のみで学習する |  |
| `264_mmecg_radarode_sceg_tau` | 264: 248(現状最良、test_corr=0.258)のODEデコーダに、原著radarODEのτ(時間遅延)を 追加した版 |  |
| `265_mmecg_radarode_sceg_control25ep` | 265: 264(τ追加)との公平な比較用control |  |
| `266_mmecg_tau_predictor_rpeak` | 266: 263と同じTauOnlyCNNアーキテクチャだが、疑似ラベルを257基準のoracle shiftではなく model-independentな`r |  |

## 267-272: radarODE公式実装忠実化・時間分解能ボトルネック解消（採用: config269系）

| config | 概要 | 判定/状態 |
|---|---|---|
| `267_mmecg_radarode_sceg_official_param` | 267: τ専用ヘッドを廃止し、公式実装(GitHub `ZYY0844/radarODE-MTL`のECGParameterEstimator)の パラメータ |  |
| `268_mmecg_radarode_sceg_anchor` | 268: 265(τなしcontrol)に、radarODE公式実装の"Anchor"補助タスク(R波位置ヒートマップ回帰、 AnchorHead)を追加する |  |
| `269_mmecg_radarode_sceg_hires` | 269: 267/268の結論(τ・公式パラメータ化・Anchor補助タスクのいずれでも改善しなかったのは、 backbone/encoderが1心拍タイトクロ |  |
| `270_mmecg_radarode_sceg_hires_80ep` | 270: 269(backbone 2段downsampling+Anchor補助タスク、25epoch)がtest_corr=0.300を達成し、 268(同 |  |
| `271_mmecg_radarode_sceg_context4s` | 271: 269/270(backbone 2段downsampling+Anchor補助タスク)に、4秒複数拍窓 (use_context_window: t |  |
| `272_mmecg_radarode_sceg_hires32` | 272: 269(タイトクロップ、backbone 2段対称downsampling、encoder_out_time=16)から、 窓の種類は変えずに時間分解 |  |

## 273-281: R波heatmap R/T混同問題の調査・恒久対応（neurokit2再ラベル、前処理比較）※2026-08-06/07

| config | 概要 | 判定/状態 |
|---|---|---|
| `273_mmecg_rpeak_spatial_gnn_neurokit_probe` | 273: 236と同一構成だが、R波の正解ラベルをneurokit2(恒久対応)で再検出する | 採用へ 8epochプローブでF1改善傾向 |
| `274_mmecg_rpeak_spatial_gnn_neurokit_full` | 274: 236(元の空間GNN, 50epoch)と同一構成だが、R波正解ラベルをneurokit2(恒久対応)で 再検出する | ★現行ベースライン F1=0.551 RR-MAE=10.69ms RMSSD-MAE=16.27ms |
| `275_mmecg_rpeak_spatial_gnn_neurokit_focal_probe` | 275: 274(neurokitラベルで学習した本番GNN, F1 0.549/RR-MAE 10.91ms)と同一構成だが、 heatmap損失をBCEから | 却下 focal lossはF1改善せず(0.285→0.503でも274未満) |
| `276_mmecg_rpeak_spatial_gnn_neurokit_rt_probe` | 276: 274(neurokitラベルで学習した本番GNN, F1 0.551/RR-MAE 10.69ms)と同一構成だが、 教師heatmapをR波単独で | プローブ段階では有望に見えたが… |
| `277_mmecg_rpeak_spatial_gnn_neurokit_rt_full` | 277: 274(R波単独教師のneurokit2本番モデル, F1 0.551/RR-MAE 10.69ms/RMSSD-MAE 16.27ms) と同一構成 | 却下 本番でF1/RMSSD-MAE悪化。原因はcorr指標によるチェックポイント選択ミスマッチ(過学習ではない)と判明 |
| `278_mmecg_rpeak_spatial_gnn_neurokit_bandpass_probe` | 278: 274(R波単独教師のneurokit2本番モデル, F1 0.551/RR-MAE 10.69ms)と同一構成だが、 RCG正規化前に[1,25]H | 却下 明確に悪化(F1=0.343) |
| `279_mmecg_rpeak_spatial_gnn_neurokit_ema_clutter_probe` | 279: 274(R波単独教師のneurokit2本番モデル)と同一構成だが、RCG正規化前にEMAベースの クラッタ除去(preprocess: ema_cl | 中立 274と大差なし(F1=0.529) |
| `280_mmecg_rpeak_spatial_gnn_neurokit_channel_weight_probe` | 280: 274(R波単独教師のneurokit2本番モデル)と同一構成だが、RCG正規化前に心拍帯パワー比 ベースのチャンネル重み付け(preprocess: | ★8epochプローブ最良 RR-MAE/RMSSD-MAE/SDNN-MAEで274本番に迫る |
| `281_mmecg_rpeak_spatial_gnn_neurokit_channel_weight_full` | 281: 280(チャネル重み付け前処理, 8epochプローブ)で有望な結果が出たため本番50epochに 進める | 評価中(2026-08-07) 本番50epoch完了、274と公平比較を実施中 |
