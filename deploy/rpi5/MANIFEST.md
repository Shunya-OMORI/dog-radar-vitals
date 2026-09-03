# Raspberry Pi 5 実機検証 エンド2エンド・マニフェスト

**状態: 準備完了(2026-08-07更新)。確定構成は config284 epoch4(sigma=15ms教師)の
空間GNN + 固定しきい値0.3 + サブサンプル重心復号 `refine_peaks_centroid`(±85ms×2回)
= F1=0.749, RR-MAE=8.41ms, RMSSD-MAE=10.27ms
(`reports/findings_2026-08-07_rpeak_heatmap_investigation.md`参照。
旧記載のconfig274+adaptive_searchbackは重心復号の導入により置き換え)。
推論エントリポイント `run_inference.py`・チェックポイント・requirements.txt を
本ディレクトリに配置済みで、WSL側のCPU 4スレッドスモークテストも通過
(trial 48: 1窓9.8ms, F1=0.940)。セットアップ手順は `README.md` 参照。
ARM対応リスク(torch-scatter等)は「torch_geometricのpure-Pythonフォールバックで
動作」を確認済みのため解消。モデル系統は空間GNNで確定。**

目的: レーダI/Q -> RCG(50点3D変位) -> R波heatmap -> R波位置、という推論パイプラインを
Raspberry Pi 5（4コアCPU、GPUなし）上で動かし、消費電力・発熱・レイテンシをエッジ
デバイス系論文向けに実測する。学習は行わない（学習はGPUワークステーション側で完結、
RPi5には学習済みチェックポイントのみ転送する）。

## 候補モデル2系統（トレードオフあり、要ユーザ判断）

| | Anchor CNN (raw-signal dilated CNN) | Spatial GNN (posXYZ + GATConv) |
|---|---|---|
| 現状精度 (F1 / RR-MAE, config274相当条件) | 旧ラベルでF1=0.593 (adaptive_searchback)。neurokitラベル本番は過学習で要再調整(下記R2参照) | **現行最良** F1=0.551 (old) / 0.687 (adaptive_searchback)、RR-MAE=10.69ms |
| 依存ライブラリ | torch のみ（軽量） | torch + **torch_geometric + torch-scatter/torch-sparse** |
| ARM/RPi5対応 | 問題なし見込み | **要注意**: torch_geometricの拡張(torch-scatter等)はARM64向けの公式prebuiltホイールが提供されないことが多く、ソースビルドが必要になる可能性が高い。事前にRPi5実機上で`pip install torch_geometric torch-scatter`を試し、ビルド時間・成否を確認しておくこと |
| パラメータ数 | 実測要 `reports/edge_efficiency_radarode_mtl.json` 参照 | 34K（本セッション内既出、極小） |

**現時点の推奨**: 精度は空間GNNが優勢だが、ARM対応リスクが未検証。RPi5実機に
到着したら**まずAnchor CNN（依存が軽い方）で疎通確認**し、時間があれば
GNN版のtorch_geometricビルドを別途試す、の二段構えを推奨する。

## 転送すべきファイル一覧

### 1. モデル定義コード（責務: model）
- Anchor CNN: `radarODE-MTL/Projects/radarODE_plus/nets/anchor_models/rpeak_dilated_cnn1d.py`
- Spatial GNN: `dog-radar-vitals/src/dog_radar_vitals/models/deep/rpeak_spatial_gnn.py`
  （`ecg_spatial_gnn.py`と共有するサブモジュールがあれば同様に; import chainを
  `python -c "import ..."` で事前確認すること）

### 2. 前処理コード（責務: preprocessing）
- `dog-radar-vitals/src/dog_radar_vitals/data/mmecg.py`（`.mat`読み込み、`MMECGRecording`）
- `dog-radar-vitals/src/dog_radar_vitals/data/mmecg_windowing.py`（`zscore_channels`のみで良い。
  channel_weight/ema_clutter/bandpassはいずれも棄却済みのため`bandpass.py`・
  `channel_weighting.py`は転送不要）

### 3. 後処理コード（責務: postprocessing）
- `dog-radar-vitals/src/dog_radar_vitals/data/rpeaks.py`
  （`extract_peaks_adaptive_searchback`が現状最良、パラメータは
  `candidate_height=0.20, threshold_frac=0.8, searchback_threshold_frac=0.6`固定）

### 4. モデルチェックポイント（責務: weights、バージョン管理必須）
- 確定後、`deploy/rpi5/checkpoints/<config番号>_<日付>.pt` の形式でコピーし、
  どのconfig・どのepoch・どのcommitで学習したかを`deploy/rpi5/checkpoints/MANIFEST.txt`
  に1行で記録する（例: `274_full epoch49 commit=<hash> F1=0.551 RR-MAE=10.69ms`）。
  runs/以下のディレクトリを直接指すのではなく、コピーを置く
  （runs/は実験のたびに増減するため転送時点のスナップショットを残す）。

### 5. テストデータ（責務: test data、個人情報に準じる取り扱い注意なし・匿名化済み91トライアル）
- `dog-radar-vitals/data/raw/mmecg/finalPartialPublicData20221108/{trial_id}.mat`
  のうちテスト被験者(17, 29, 30)のトライアルのみで十分（全91トライアルを転送する必要はない）。
  `dog_radar_vitals.data.mmecg.trial_ids_for_subjects`で対象trial_idを列挙できる。

### 6. 推論スクリプト（責務: entrypoint、RPi5専用に新規作成が必要）
- 既存の`evaluate_spatial_gnn_hrv.py`/`evaluate_anchor_edge_online.py`はGPU前提の
  評価用でHRV計算等の付随処理が多い。RPi5では**新規に`deploy/rpi5/run_inference.py`を
  作成し**、(1)電力・温度ロギング開始、(2)1トライアル分のRCG読み込み、(3)前処理、
  (4)スライディング窓推論、(5)後処理でR波位置抽出、(6)電力・温度ロギング終了・レポート出力、
  のみに絞った最小構成にすること（このファイルは未作成、要実装）。
- 参考実装: `dog-radar-vitals/scripts/benchmark_edge_efficiency_gnn.py`と
  `radarODE-MTL/scripts/benchmark_edge_efficiency.py`が既に「CPU 4スレッド固定
  （RPi5相当のプロキシ）」でレイテンシ計測をしている。RPi5実機ではこれらのGPU電力計測
  (`nvidia-smi`)部分を、RPi5の電力計測手段（USB電力計 or `vcgencmd measure_temp`+
  外部電力計など、計測手法は要相談）に置き換える形で流用できる。

## 必要ライブラリ（RPi5側、最小構成）

`dog-radar-vitals/pyproject.toml`のdependenciesは学習・実験全体を含み過大（transformers,
accelerate, bitsandbytes等は今回の推論パイプラインには不要）。RPi5には以下のみで足りるはず:

```
torch>=2.2          # ARM64向けCPUビルドの入手性を事前確認(公式wheel or piwheels)
numpy>=1.24
scipy>=1.10          # channel_weighting.pyのscipy.signal.welch用（採用時のみ）
neurokit2>=0.2.7     # 正解ラベル生成が必要な場合のみ(推論自体には不要、精度検証用)
```

Spatial GNNを使う場合のみ追加:
```
torch_geometric>=2.5
torch-scatter        # ARM64ビルド未検証、要事前確認
torch-sparse          # 同上
```

## 未確定事項・要フォローアップ
1. どちらのモデル系統を送るか（精度 vs ARM対応リスクのトレードオフ、上記参照）
2. `deploy/rpi5/run_inference.py`は未実装（骨子のみ本ファイルに記載）
3. 電力・温度の実測手段（USB電力計の型番、`vcgencmd`等）は未確定、実機到着後に決める
