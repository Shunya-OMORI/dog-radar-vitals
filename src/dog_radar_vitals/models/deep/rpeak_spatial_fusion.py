"""空間融合モデルのheatmap版。`ecg_spatial_fusion.py`と入力・バックボーンは同じだが、
出力層にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.ecg_spatial_fusion import PerPointTemporalEncoder, TemporalDecoder


class RPeakSpatialFusion(nn.Module):
    def __init__(self, n_points: int = 50, embed_dim: int = 32, n_heads: int = 4, n_attn_layers: int = 2) -> None:
        super().__init__()
        self.point_encoder = PerPointTemporalEncoder(embed_dim)
        self.pos_embed = nn.Linear(3, embed_dim)

        attn_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2, dropout=0.1, batch_first=True
        )
        self.spatial_attn = nn.TransformerEncoder(attn_layer, num_layers=n_attn_layers)
        self.decoder = TemporalDecoder(embed_dim)

    def forward(self, rcg: torch.Tensor, posxyz: torch.Tensor) -> torch.Tensor:
        """rcg: (batch, seq_len, n_points)。posxyz: (batch, n_points, 3)。-> (batch, seq_len)、値域[0,1]。"""
        batch, seq_len, n_points = rcg.shape
        x = rcg.transpose(1, 2)

        point_feat = self.point_encoder(x)
        t_reduced = point_feat.shape[-1]

        pos_emb = self.pos_embed(posxyz)
        point_feat = point_feat + pos_emb.unsqueeze(-1)

        h = point_feat.permute(0, 3, 1, 2).reshape(batch * t_reduced, n_points, -1)
        h = self.spatial_attn(h)
        h = h.mean(dim=1)
        h = h.reshape(batch, t_reduced, -1).transpose(1, 2)

        return torch.sigmoid(self.decoder(h, target_len=seq_len))
