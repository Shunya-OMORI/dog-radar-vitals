"""レーダのanalytic signal(複素数)からECG波形を推定する複素畳み込みCNN（本命の1つ）。

`ecg_cnn1d.py`と同じ dilation を2倍ずつ増やす全畳み込みバックボーン構成だが、
`nn.Conv1d(..., dtype=torch.complex64)`で複素畳み込みを行い、活性化に`ModReLU`
（`complex_layers.py`）を使う点が異なる。ECGの真値は実数波形なので、最終層で
複素特徴量の実部・虚部を連結してから通常の実数`Conv1d`で1chに落とす
（複素→実数のreadout）。入力は`data/mmecg_windowing.py`の`complex_input=True`で
Hilbert変換したanalytic signal（各チャネルの位相情報を保持したもの）を想定する。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.complex_layers import ModReLU


class ECGWaveformComplexCNN1D(nn.Module):
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
        self.head = nn.Conv1d(channels * 2, 1, kernel_size=1)  # 複素(re,im)を連結して実数へ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) complex64 -> (batch, seq_len) float"""
        h = x.transpose(1, 2)  # (batch, in_channels, seq_len) complex
        h = self.conv(h)
        h_ri = torch.cat([h.real, h.imag], dim=1)  # (batch, channels*2, seq_len) float
        out = self.head(h_ri)
        return out.squeeze(1)
