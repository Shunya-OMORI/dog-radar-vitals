"""レーダI/Q波形からR波heatmap（疎なイベント検出）を推定する1次元CNN。

`ecg_cnn1d.py`（密な波形振幅を回帰、出力に活性化なし）と入力・バックボーンは同じだが、
出力層にsigmoidを通し[0,1]のheatmap値として解釈する点が異なる。これにより損失も
MSEではなくBCE（二値のようなガウシアン目標に対する交差エントロピー）を使うのが自然になり、
「密な波形再構成」ではなく「疎なイベントのタイミング検出」という別のタスク定式化になる。
"""
from __future__ import annotations

import torch
from torch import nn


class RPeakCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 64,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_layers):
            dilation = 2**i
            layers += [
                nn.Conv1d(in_ch, channels, kernel_size, padding="same", dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 2) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h = x.transpose(1, 2)
        h = self.conv(h)
        out = self.head(h)
        return torch.sigmoid(out).squeeze(1)
