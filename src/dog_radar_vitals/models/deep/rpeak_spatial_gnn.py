"""空間GNNモデルのheatmap版。`ecg_spatial_gnn.py`と入力・バックボーンは同じだが、出力層に
sigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv

from dog_radar_vitals.models.deep.ecg_spatial_fusion import PerPointTemporalEncoder, TemporalDecoder
from dog_radar_vitals.models.deep.spatial_graph import batched_knn_edge_index


class RPeakSpatialGNN(nn.Module):
    def __init__(self, n_points: int = 50, embed_dim: int = 32, k_neighbors: int = 6, n_gat_layers: int = 2,
                 n_downsample: int = 3, spatial_aggregation: str = "knn_gat") -> None:
        """n_downsample (2026-08-05, default 3 = unchanged/8x): F1 for this
        model (best RR-interval accuracy this session, 9.67ms) has stayed
        moderate (0.383); PerPointTemporalEncoder's 8x downsampling is the
        same class of temporal-resolution bottleneck already diagnosed as
        the root cause of the Anchor task's earlier localization/recall
        problems. Exposed here to test whether less downsampling (e.g. 1 =
        2x) improves F1 without hurting RR-MAE."""
        super().__init__()
        self.point_encoder = PerPointTemporalEncoder(embed_dim, n_downsample=n_downsample)
        self.pos_embed = nn.Linear(3, embed_dim)
        self.k_neighbors = k_neighbors

        # spatial_aggregation (2026-08-24 追加, 既定は従来と同じ knn_gat):
        #   「なぜグラフなのか」に対照実験で答えるための切り替え。点ごとエンコーダ・
        #   位置埋め込み・時間デコーダは一切変えず、**50点をどう混ぜるかだけ**を変える。
        #     none      : 混ぜない(このあとの平均プーリングだけ)
        #     knn_gat   : 3次元距離のk近傍グラフ上で注意つきグラフ畳み込み(採用構成)
        #     full_attn : 全点どうしの自己注意(グラフを張らず全結合にした場合)
        self.spatial_aggregation = spatial_aggregation
        if spatial_aggregation == "knn_gat":
            self.gat_layers = nn.ModuleList(
                [GATConv(embed_dim, embed_dim, heads=1) for _ in range(n_gat_layers)]
            )
        elif spatial_aggregation == "full_attn":
            self.attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
                 for _ in range(n_gat_layers)]
            )
        elif spatial_aggregation != "none":
            raise ValueError(f"unknown spatial_aggregation '{spatial_aggregation}'")
        self.decoder = TemporalDecoder(embed_dim)

    def forward(self, rcg: torch.Tensor, posxyz: torch.Tensor) -> torch.Tensor:
        """rcg: (batch, seq_len, n_points)。posxyz: (batch, n_points, 3)。-> (batch, seq_len)、値域[0,1]。"""
        batch, seq_len, n_points = rcg.shape
        x = rcg.transpose(1, 2)

        point_feat = self.point_encoder(x)
        t_reduced = point_feat.shape[-1]

        pos_emb = self.pos_embed(posxyz)
        point_feat = point_feat + pos_emb.unsqueeze(-1)

        h = point_feat.permute(0, 3, 1, 2).reshape(batch * t_reduced * n_points, -1)

        if self.spatial_aggregation == "knn_gat":
            edge_index = batched_knn_edge_index(posxyz, self.k_neighbors, n_repeat=t_reduced).to(rcg.device)
            for gat in self.gat_layers:
                h = torch.relu(gat(h, edge_index))
        elif self.spatial_aggregation == "full_attn":
            h = h.reshape(batch * t_reduced, n_points, -1)
            for attn in self.attn_layers:
                h = torch.relu(attn(h, h, h, need_weights=False)[0])
            h = h.reshape(batch * t_reduced * n_points, -1)

        h = h.reshape(batch, t_reduced, n_points, -1).mean(dim=2)
        h = h.transpose(1, 2)

        return torch.sigmoid(self.decoder(h, target_len=seq_len))
