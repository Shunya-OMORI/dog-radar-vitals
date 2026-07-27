"""レーダ波形からR波heatmapを推定するConformer型モデル。`ecg_conformer.py`と入力・バックボーンは
同じだが、出力層にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.ecg_conformer import ConvSubsampler
from dog_radar_vitals.models.deep.transformer import SinusoidalPositionalEncoding


class RPeakConformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 50,
        d_model: int = 128,
        conv_kernel_size: int = 15,
        conv_layers: int = 3,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.local_conv = ConvSubsampler(in_channels, d_model, conv_kernel_size, conv_layers)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h = self.local_conv(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return torch.sigmoid(self.head(h).squeeze(-1))
