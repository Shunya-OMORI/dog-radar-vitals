"""複素領域モデル v2のheatmap版。`ecg_complex_cnn_v2.py`と入力・バックボーンは同じだが、
出力層にsigmoidを通す点が異なる（`ecg_cnn1d.py`/`rpeak_cnn1d.py`の対応関係と同じ）。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.ecg_complex_cnn_v2 import ComplexResidualBlock


class RPeakComplexCNN1DV2(nn.Module):
    def __init__(
        self,
        in_channels: int = 50,
        channels: int = 32,
        n_blocks: int = 6,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, channels, kernel_size=1, dtype=torch.complex64)
        self.blocks = nn.ModuleList(
            [ComplexResidualBlock(channels, kernel_size, dilation=2**i) for i in range(n_blocks)]
        )
        self.head = nn.Conv1d(channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) complex64 -> (batch, seq_len) float、値域[0,1]。"""
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h_ri = torch.cat([h.real, h.imag], dim=1)
        return torch.sigmoid(self.head(h_ri).squeeze(1))
