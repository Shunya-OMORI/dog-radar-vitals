# ecg_*（波形推定）で非自明な成果を挙げた3アーキテクチャ

対象: mmWave radar (RCG, 50ch) → ECG waveform regression, single split (train 7 / val 1 / test 3 subjects), loss=`nn.MSELoss()`共通。指標はtest全体平均のPearson相関・RR Interval MAE。baseline (`ecg_cnn1d`, dilated conv1d stack): corr=0.224, RR MAE=21.2ms。

## 1. `ecg_unet1d` — corr=0.209, RR MAE=**15.9ms（全ecg_*中最良、baseline比-25%）**

**狙い**: QRS(鋭い・高周波)とP/T波(緩やか・低周波)を異なる時間解像度で捉える。
**根拠**: LifWavNet (arXiv:2510.27692) の「lifting waveletによる多重解像度分解がRF→ECG再構成を大きく改善する」という報告。本式のlifting scheme実装はコストが高いため、同じ設計思想を標準U-Netで代替。
**成立条件**: 波形回帰(heatmap版`rpeak_unet1d`は20.4msでbaseline比未改善)。スキップ接続によるサイズ不一致は`F.interpolate`で吸収（seq_len=800は2^3で割り切れるため通常は発生しない）。

```python
skips = []
for enc_block, down in zip(self.enc_blocks, self.downs):
    h = enc_block(h); skips.append(h); h = down(h)
h = self.bottleneck(h)
for up, dec_block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
    h = up(h)
    if h.shape[-1] != skip.shape[-1]:
        h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
    h = dec_block(torch.cat([h, skip], dim=1))  # channel-wise skip concat
```

## 2. `ecg_spatial_gnn` — corr=0.188, RR MAE=17.9ms（**パラメータ33,985、全モデル中最少級**）

**狙い**: 50点のRCGを「3D座標を持つ点群」として扱い、実距離が近い点同士だけを融合する。
**根拠**: Chen et al. 2022 (RCG2ECG、本データセット原著)のposXYZ位置埋め込み+Transformer空間融合、およびMMPoint-GNN (Gong et al. 2021, mmWave point cloud HAR)のグラフ構造化。全結合Self-Attention版(`ecg_spatial_fusion`)より**少ないパラメータで上回った**点が非自明。
**成立条件**: `torch_geometric`の`knn_graph`は`pyg-lib`依存で本環境未導入のため、`torch.cdist`+`topk`で自前実装。posXYZはトライアルごとに異なる実測値（`data/mmecg.py`の`MMECGRecording.posxyz`、固定グリッドではない）。

```python
dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
dist.fill_diagonal_(float("inf"))
_, nn_idx = torch.topk(dist, k, dim=1, largest=False)  # k近傍
# GATConv (torch_geometric.nn) でメッセージパッシング
h = torch.relu(gat(h, edge_index))  # h: (batch*T'*n_points, embed_dim)
```

## 3. `ecg_conv_ncp` — corr=0.210, RR MAE=**19.5ms（素のecg_ncp比 21.1→19.5ms, F1 0.311→0.400）**

**狙い**: NCP(`ncps.torch.CfC`)は連続時間・低域通過的なダイナミクスを持ち生信号のノイズに弱い（別実験で実証: ノイズ注入時の相関劣化率がCNN/U-Netより急峻）。局所畳み込みで事前に形状特徴を抽出しノイズを平滑化する。
**根拠**: 本セッション内でConformer（局所conv+Self-Attention）が素のTransformerの未収束（rpeak_transformer, F1=0）を解消した知見をNCPに転用。
**成立条件**: `mixed_memory=True`（LSTM様メモリセル併用）必須、2層CfCスタック。**代償: CPU推論レイテンシ278ms、baseline(2.0ms)の139倍**——パラメータ数(137,933)は小さくてもFLOPs効率がレイテンシに直結しない（CfCが時間方向に逐次実行されるため）。

```python
from ncps.torch import CfC
from ncps.wirings import AutoNCP
h = self.local_conv(x)          # QRS幅程度の受容野を持つdilated conv (ConvSubsampler)
for ncp in self.ncp_layers:     # wiring = AutoNCP(units, out_dim); CfC(in_dim, wiring, ...)
    h, _ = ncp(h)                # (batch, seq_len, out_dim), mixed_memory=Trueでhxはtuple
```

---
出典: 詳細な数値・実験ログは`EXPERIMENTS.md`、先行研究の書誌情報は`prior_work_accuracy_comparison.md`参照。
