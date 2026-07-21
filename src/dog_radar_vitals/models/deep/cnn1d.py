"""レンジビンをチャネルとして扱う1次元CNN。Transformerより軽量な比較対象。"""
from __future__ import annotations

import torch
from torch import nn


class VitalsCNN1D(nn.Module):
    def __init__(
        self,
        n_bins: int = 467,
        channels: tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 7,
        dropout: float = 0.1,
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = n_bins
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_bins) -> (batch, n_outputs)"""
        h = x.transpose(1, 2)  # (batch, n_bins, seq_len)
        h = self.conv(h)
        pooled = h.mean(dim=-1)
        return self.head(pooled)
