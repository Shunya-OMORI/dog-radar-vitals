"""レーダI/Q波形からECG波形を推定する全畳み込み1次元CNN（sequence-to-sequence）。

HR/BR予測（`cnn1d.py`, `transformer.py`等）が「窓 -> スカラ1個」の回帰なのに対し、
こちらは「窓 -> 同じ長さの波形」を出力する点が異なる。ダウンサンプリングを行わず
`padding='same'` で全層とも時間長を保つため、出力は入力と厳密に同じ長さになる。
受容野を広げるため層ごとにdilationを2倍ずつ増やす。
"""
from __future__ import annotations

import torch
from torch import nn


class ECGWaveformCNN1D(nn.Module):
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

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x: (batch, seq_len, 2) -> (batch, seq_len)。

        `return_features=True`で、headに入れる前の中間特徴(batch, channels, seq_len)も
        併せて返す(`TemporalShiftHead`のlocalization networkの入力に使う)。
        """
        h = x.transpose(1, 2)  # (batch, 2, seq_len)
        h = self.conv(h)
        out = self.head(h).squeeze(1)  # (batch, seq_len)
        if return_features:
            return out, h
        return out
