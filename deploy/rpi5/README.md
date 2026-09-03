# Raspberry Pi 5 推論パッケージ

確定構成（2026-08-07）の推論を Raspberry Pi 5（4コアCPU・GPUなし）で動かすための一式。
接続・環境の全体計画は `reports/mmecg_comparison_2026-08-05/raspberrypi5_plan.md`
（Tailscale + SSH で WSL2 の Claude Code から `ssh raspi` 操作できるようにする）を参照。

## 構成

- モデル: 空間GNN `RPeakSpatialGNN`（0.034 M パラメータ）
- チェックポイント: `checkpoints/284_sigma15_epoch004.pt`（`checkpoints/MANIFEST.txt` 参照）
- 後処理: 固定しきい値 0.3 → サブサンプル重心復号（±85 ms soft-argmax，2回反復）
- WSL側での確認済み性能: F1 0.749 / RR-MAE 8.41 ms / RMSSD-MAE 10.27 ms（test 44トライアル）

## Pi 上でのセットアップ（`ssh raspi` 経由，Claude Code 自動化想定）

```bash
# 1. リポジトリを転送（データは重いのでテスト被験者のトライアルのみで良い）
#    rsync -av --exclude .venv --exclude runs --exclude data dog-radar-vitals/ raspi:~/dog-radar-vitals/
#    rsync -av dog-radar-vitals/data/raw/mmecg/finalPartialPublicData20221108/48.mat raspi:~/dog-radar-vitals/data/raw/mmecg/finalPartialPublicData20221108/

# 2. venv + 依存（torch はCPU版。全部で ~500MB 程度）
python3 -m venv ~/dog-radar-vitals/.venv
~/dog-radar-vitals/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/dog-radar-vitals/.venv/bin/pip install -r ~/dog-radar-vitals/deploy/rpi5/requirements.txt

# 3. 実行（--no-gt なら neurokit2 も不要）
cd ~/dog-radar-vitals && .venv/bin/python deploy/rpi5/run_inference.py --trial 48 --threads 4
```

torch-scatter / torch-sparse などのコンパイル拡張は**不要**（torch_geometric の
pure-Python フォールバックで GATConv が動くことを WSL 側で確認済み。ARM ソース
ビルドのリスクは解消）。

## WSL 側スモークテスト結果（CPU 4スレッド固定 = Pi 相当のプロキシ，2026-08-07）

```
trial 48: 178s, 695 windows (step 0.25s)
inference per 4s-window: mean 9.8 ms (p95 12.4 ms), occupancy 3.9% (step 0.25s)
F1=0.940 RR-MAE=8.61 ms RMSSD err=0.76 ms (radar 32.64 vs ECG 33.41)
```

x86 の 4 スレッドは Pi 5 より速いため実機では数倍のレイテンシを見込む。それでも
0.25 秒ステップの実時間処理に対して 1 桁以上の余裕がある見込み。

## 実機で測るもの（残課題）

- 1窓あたりレイテンシ（mean/p95）と実時間占有率
- 消費電力・温度（USB電力計 or `vcgencmd measure_temp`，手段は未確定）
