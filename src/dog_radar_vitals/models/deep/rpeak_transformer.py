"""レーダ波形からR波heatmapを推定するTransformer Encoder。`ecg_transformer.py`と入力・
バックボーンは同じだが、出力層にsigmoidを通す点が異なる（`rpeak_cnn1d.py`との対応関係と同じ）。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.transformer import SinusoidalPositionalEncoding


class RPeakTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return torch.sigmoid(self.head(h).squeeze(-1))
