"""RCGの50点を、posXYZの実距離に基づくk近傍グラフとして扱うGNNモデル（本命3の追加派生）。

`ecg_spatial_fusion.py`（50点全結合のSelf-Attention）に対し、本モデルは
`spatial_graph.knn_edge_index`で構築した**実際の3D距離によるk近傍グラフ**上で
`torch_geometric.nn.GATConv`によるメッセージパッシングを行う。MMPoint-GNN(Gong et al. 2021)
等のmmWave point cloud向けGNN研究に倣い、「空間的に近い点同士が情報をやり取りする」という
物理的に妥当な帰納バイアスを、全結合Self-Attentionより明示的に組み込む。
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv

from dog_radar_vitals.models.deep.ecg_spatial_fusion import PerPointTemporalEncoder, TemporalDecoder
from dog_radar_vitals.models.deep.spatial_graph import batched_knn_edge_index


class ECGSpatialGNN(nn.Module):
    def __init__(self, n_points: int = 50, embed_dim: int = 32, k_neighbors: int = 6, n_gat_layers: int = 2) -> None:
        super().__init__()
        self.point_encoder = PerPointTemporalEncoder(embed_dim)
        self.pos_embed = nn.Linear(3, embed_dim)
        self.k_neighbors = k_neighbors

        self.gat_layers = nn.ModuleList(
            [GATConv(embed_dim, embed_dim, heads=1) for _ in range(n_gat_layers)]
        )
        self.decoder = TemporalDecoder(embed_dim)

    def forward(self, rcg: torch.Tensor, posxyz: torch.Tensor) -> torch.Tensor:
        """rcg: (batch, seq_len, n_points)。posxyz: (batch, n_points, 3)。-> (batch, seq_len)"""
        batch, seq_len, n_points = rcg.shape
        x = rcg.transpose(1, 2)

        point_feat = self.point_encoder(x)  # (batch, n_points, embed_dim, T')
        t_reduced = point_feat.shape[-1]

        pos_emb = self.pos_embed(posxyz)  # (batch, n_points, embed_dim)
        point_feat = point_feat + pos_emb.unsqueeze(-1)

        # (batch, n_points, embed_dim, T') -> (batch*T'*n_points, embed_dim)。ノード順はbatch*T'個の
        # グラフが連続、というbatched_knn_edge_indexの前提と合わせる。
        h = point_feat.permute(0, 3, 1, 2).reshape(batch * t_reduced * n_points, -1)

        edge_index = batched_knn_edge_index(posxyz, self.k_neighbors, n_repeat=t_reduced).to(rcg.device)
        for gat in self.gat_layers:
            h = torch.relu(gat(h, edge_index))

        h = h.reshape(batch, t_reduced, n_points, -1).mean(dim=2)  # 点方向に平均プーリング
        h = h.transpose(1, 2)  # (batch, embed_dim, T')

        return self.decoder(h, target_len=seq_len)
