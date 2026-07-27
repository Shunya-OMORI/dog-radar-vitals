"""レーダ波形からR波heatmapを推定するNCPモデル。`ecg_ncp.py`と入力・配線は同じだが、
出力にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn


class RPeakNCP(nn.Module):
    def __init__(self, in_channels: int = 2, units: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        wiring = AutoNCP(units, output_dim)
        self.rnn = CfC(in_channels, wiring, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h, _ = self.rnn(x)
        return torch.sigmoid(h.squeeze(-1))
