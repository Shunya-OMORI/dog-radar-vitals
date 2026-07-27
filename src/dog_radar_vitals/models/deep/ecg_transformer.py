"""レーダ波形からECG波形を推定するTransformer Encoder（sequence-to-sequence）。

犬用`models/deep/transformer.py`の`SinusoidalPositionalEncoding`を再利用するが、
あちらは時系列を平均プーリングしてスカラ回帰するのに対し、こちらはプーリングせず
各時刻の出力に線形headを適用して入力と同じ長さの波形を出力する点が異なる
（`ecg_cnn1d.py`のTransformer版、通常DNNベースラインとして追加）。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.transformer import SinusoidalPositionalEncoding


class ECGWaveformTransformer(nn.Module):
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
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)
