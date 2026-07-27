"""レーダ波形からECG波形を推定する双方向LSTM（sequence-to-sequence、通常DNNベースライン）。"""
from __future__ import annotations

import torch
from torch import nn


class ECGWaveformLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)
