"""複素領域モデル v2: Complex Batch Normalization + 複素残差接続を追加した深い複素畳み込みCNN。

v1(`ecg_complex_cnn.py`)は正規化層を持たなかったため、層を深くすると学習が不安定になりやすい
制約があった。v2は`complex_layers.ComplexBatchNorm1d`（Trabelsi et al. 2018）を各層に挟み、
さらに複素の残差接続（`h + block(h)`は複素数のまま定義できる）を追加することで、v1より
深いネットワークを安定して学習できるようにする。

設計の数学的根拠: analytic signal（Hilbert変換）の瞬時位相は、胸壁変位が小さい範囲では
変位量に線形近似できる（レーダのarctangent復調と同じ理論的根拠）。複素ネットワークは
この位相情報を実数ネットワークのように振幅へ潰さずに保持できるが、深いネットワークでは
勾配が不安定になりやすい。Complex BNによる白色化と残差接続が、この安定化を担う。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.complex_layers import ComplexBatchNorm1d, ModReLU


class ComplexResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation, dtype=torch.complex64)
        self.bn1 = ComplexBatchNorm1d(channels)
        self.act1 = ModReLU(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding="same", dilation=dilation, dtype=torch.complex64)
        self.bn2 = ComplexBatchNorm1d(channels)
        self.act2 = ModReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act2(x + h)


class ECGWaveformComplexCNN1DV2(nn.Module):
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
        self.head = nn.Conv1d(channels * 2, 1, kernel_size=1)  # 複素(re,im)を連結して実数へ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) complex64 -> (batch, seq_len) float"""
        h = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h_ri = torch.cat([h.real, h.imag], dim=1)
        return self.head(h_ri).squeeze(1)
