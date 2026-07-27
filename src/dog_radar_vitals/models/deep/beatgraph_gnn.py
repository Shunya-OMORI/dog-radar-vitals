"""RCG beat segment -> PQRST 5点グラフ(時刻オフセット, 振幅) を回帰するGNNモデル（本命の1つ）。

拍単位の波形をP/Q/R/S/T 5点のグラフとして扱うというユーザの発想を実装したもの。
1D CNNでレーダのbeat segment(50ch)をグローバル文脈ベクトルにエンコードし、
5個のlearnableノード初期埋め込み+グローバル文脈を`torch_geometric.nn.GATConv`で
数層メッセージパッシングして各ノードの(時刻オフセット, 振幅)を回帰する。

全サンプルでグラフ構造(P-Q-R-S-T鎖状、無向)が共通なため、torch_geometricの`Batch`は使わず、
バッチ内のB個のグラフをノードID オフセットで手動結合した1つの大きなグラフとして
GATConvに渡す（B*5ノードの疎グラフ、標準的なグラフバッチング手法）。
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv

# P-Q-R-S-T の鎖状エッジ（無向 = 両方向）。ノード順はdata/mmecg_beatgraph_dataset.pyの
# BEAT_GRAPH_NODE_ORDER (P=0,Q=1,R=2,S=3,T=4) と揃える。
_CHAIN_EDGES = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
N_NODES = 5
N_NODE_FEATURES = 2  # (time_offset_norm, amplitude)


class RadarBeatEncoder(nn.Module):
    """RCG beat segment(50ch) -> グローバル文脈ベクトル。"""

    def __init__(self, in_channels: int, embed_dim: int, channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding="same"),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding="same", dilation=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, embed_dim)"""
        h = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.proj(h)


def _batched_chain_edge_index(batch_size: int, device: torch.device) -> torch.Tensor:
    edges = _CHAIN_EDGES.to(device)
    offsets = (torch.arange(batch_size, device=device) * N_NODES).repeat_interleave(edges.shape[1])
    tiled = edges.repeat(1, batch_size)
    return tiled + offsets.unsqueeze(0)


class BeatGraphGNN(nn.Module):
    def __init__(self, in_channels: int = 50, embed_dim: int = 32, gat_hidden: int = 32, n_gat_layers: int = 3) -> None:
        super().__init__()
        self.encoder = RadarBeatEncoder(in_channels, embed_dim)
        self.node_init = nn.Parameter(torch.randn(N_NODES, embed_dim) * 0.1)

        gat_layers = []
        in_dim = embed_dim
        for i in range(n_gat_layers):
            out_dim = gat_hidden if i < n_gat_layers - 1 else embed_dim
            gat_layers.append(GATConv(in_dim, out_dim))
            in_dim = out_dim
        self.gat_layers = nn.ModuleList(gat_layers)
        self.head = nn.Linear(embed_dim, N_NODE_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, 5, 2)"""
        batch_size = x.shape[0]
        global_ctx = self.encoder(x)  # (batch, embed_dim)

        node_x = self.node_init.unsqueeze(0) + global_ctx.unsqueeze(1)  # (batch, 5, embed_dim)
        node_x = node_x.reshape(batch_size * N_NODES, -1)  # (batch*5, embed_dim)

        edge_index = _batched_chain_edge_index(batch_size, x.device)
        for gat in self.gat_layers:
            node_x = torch.relu(gat(node_x, edge_index))

        out = self.head(node_x)  # (batch*5, 2)
        return out.reshape(batch_size, N_NODES, N_NODE_FEATURES)
