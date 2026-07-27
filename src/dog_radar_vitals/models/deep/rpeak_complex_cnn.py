"""レーダのanalytic signal(複素数)からR波heatmapを推定する複素畳み込みCNN。
`ecg_complex_cnn.py`と入力・バックボーンは同じだが、出力層にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.complex_layers import ModReLU


class RPeakComplexCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 50,
        channels: int = 32,
        n_layers: int = 6,
        kernel_size: int = 15,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_layers):
            dilation = 2**i
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size, padding="same", dilation=dilation, dtype=torch.complex64),
                ModReLU(channels),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) complex64 -> (batch, seq_len) float、値域[0,1]。"""
        h = x.transpose(1, 2)
        h = self.conv(h)
        h_ri = torch.cat([h.real, h.imag], dim=1)
        out = self.head(h_ri)
        return torch.sigmoid(out.squeeze(1))
