"""レーダ波形からR波heatmapを推定する双方向LSTM。`ecg_lstm.py`と入力・バックボーンは同じだが、
出力層にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from torch import nn


class RPeakLSTM(nn.Module):
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
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h, _ = self.lstm(x)
        return torch.sigmoid(self.head(h).squeeze(-1))
