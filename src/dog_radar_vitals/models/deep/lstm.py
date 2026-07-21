"""レンジビン系列を再帰的に処理するLSTMベースの回帰モデル。"""
from __future__ import annotations

import torch
from torch import nn


class VitalsLSTM(nn.Module):
    def __init__(
        self,
        n_bins: int = 467,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_bins,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_bins) -> (batch, n_outputs)"""
        h, _ = self.lstm(x)
        pooled = h.mean(dim=1)
        return self.head(pooled)
