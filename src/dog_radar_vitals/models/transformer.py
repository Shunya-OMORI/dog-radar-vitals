"""レーダのレンジビン系列からバイタルサインを回帰する素朴なTransformer Encoder。

各時刻のレンジビンベクトル(467次元)をトークンとして扱い、Transformer Encoderで
系列全体を処理したのち平均プーリングして回帰する。将来のマルチタスク学習
（心拍数・呼吸数の同時予測）に備え、出力次元 `n_outputs` は複数に拡張できる。
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class VitalsTransformer(nn.Module):
    def __init__(
        self,
        n_bins: int = 467,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_bins, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_bins) -> (batch, n_outputs)"""
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)
