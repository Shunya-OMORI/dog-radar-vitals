"""RCGの50点を「3D空間分布を持つ点群」として扱う空間融合モデル（本命モデルの追加候補）。

Chen et al. (2022, RCG2ECG、同一データセットの原著論文)のアーキテクチャは、50点の心臓動き
計測を単なる畳み込みチャネルとしてではなく、各点の3D位置(posXYZ)を埋め込んでTransformerで
時間特徴と融合している（Fig.6「Spatial feature」ブランチ）。既存の`ecg_cnn1d`等は
posXYZを一切使わず、50chを通常の畳み込みチャネルとして扱っている。本モデルは以下の設計で
posXYZを明示的に活用する:

1. 各点を独立に(重み共有)1D convで時間方向にダウンサンプリングし、点ごとの時間特徴を得る
   （全点が同一の物理量=胸壁変位を測っているため、重み共有が妥当という設計判断）。
2. posXYZの線形埋め込みを各点の特徴に加算する（Transformerの位置エンコーディングと同じ発想）。
3. 各時刻について50点間でSelf-Attentionを行い、空間的な融合（近い点・関連する点への
   注目）を学習する（時間方向で重みを共有、計算量を抑える簡略化）。
4. 点方向に平均プーリングしたのち、転置畳み込みで時間解像度を戻し波形を出力する。
"""
from __future__ import annotations

import torch
from torch import nn


class PerPointTemporalEncoder(nn.Module):
    """50点すべてに重み共有の1D convを適用し、時間方向にダウンサンプリングする。"""

    def __init__(self, embed_dim: int, n_downsample: int = 3, kernel_size: int = 9) -> None:
        super().__init__()
        layers = []
        in_ch = 1
        for _ in range(n_downsample):
            layers += [
                nn.Conv1d(in_ch, embed_dim, kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            ]
            in_ch = embed_dim
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_points, T) -> (batch, n_points, embed_dim, T')"""
        batch, n_points, t = x.shape
        h = x.reshape(batch * n_points, 1, t)
        h = self.conv(h)
        return h.reshape(batch, n_points, h.shape[1], h.shape[2])


class TemporalDecoder(nn.Module):
    """点方向にプーリングした融合特徴を、元の時間解像度まで転置畳み込みで戻す。"""

    def __init__(self, embed_dim: int, n_upsample: int = 3, kernel_size: int = 9) -> None:
        super().__init__()
        layers = []
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
            ]
        self.deconv = nn.Sequential(*layers)
        self.head = nn.Conv1d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        h = self.deconv(x)
        if h.shape[-1] != target_len:
            h = nn.functional.interpolate(h, size=target_len, mode="linear", align_corners=False)
        return self.head(h).squeeze(1)


class ECGSpatialFusion(nn.Module):
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
        """rcg: (batch, seq_len, n_points)。posxyz: (batch, n_points, 3)。-> (batch, seq_len)"""
        batch, seq_len, n_points = rcg.shape
        x = rcg.transpose(1, 2)  # (batch, n_points, seq_len)

        point_feat = self.point_encoder(x)  # (batch, n_points, embed_dim, T')
        t_reduced = point_feat.shape[-1]

        pos_emb = self.pos_embed(posxyz)  # (batch, n_points, embed_dim)
        point_feat = point_feat + pos_emb.unsqueeze(-1)

        # 時間方向で重み共有: (batch, n_points, embed_dim, T') -> (batch*T', n_points, embed_dim)
        h = point_feat.permute(0, 3, 1, 2).reshape(batch * t_reduced, n_points, -1)
        h = self.spatial_attn(h)
        h = h.mean(dim=1)  # 点方向に平均プーリング -> (batch*T', embed_dim)
        h = h.reshape(batch, t_reduced, -1).transpose(1, 2)  # (batch, embed_dim, T')

        return self.decoder(h, target_len=seq_len)
