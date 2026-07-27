"""NCP進化系のheatmap版。`ecg_conv_ncp.py`と入力・バックボーンは同じだが、出力層に
sigmoidを通す点が異なる。207/208フェーズでrpeak_ncpが崩壊した（LTC/CfCの連続時間ダイナミクスが
heatmapの鋭いスパイクに構造的に不向きという仮説）ため、局所畳み込みで事前にエッジ検出的な
特徴を与えることで、この崩壊を解消できるかを検証する。
"""
from __future__ import annotations

import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn

from dog_radar_vitals.models.deep.ecg_conformer import ConvSubsampler


class RPeakConvNCP(nn.Module):
    def __init__(
        self,
        in_channels: int = 50,
        conv_channels: int = 32,
        conv_kernel_size: int = 15,
        conv_layers: int = 3,
        ncp_units: int = 64,
        ncp_output_dim: int = 8,
        n_ncp_layers: int = 2,
        mixed_memory: bool = True,
    ) -> None:
        super().__init__()
        self.local_conv = ConvSubsampler(in_channels, conv_channels, conv_kernel_size, conv_layers)

        self.ncp_layers = nn.ModuleList()
        in_dim = conv_channels
        for i in range(n_ncp_layers):
            out_dim = ncp_output_dim if i == n_ncp_layers - 1 else ncp_units // 2
            wiring = AutoNCP(ncp_units, out_dim)
            self.ncp_layers.append(CfC(in_dim, wiring, batch_first=True, mixed_memory=mixed_memory))
            in_dim = out_dim

        self.head = nn.Linear(ncp_output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h = self.local_conv(x)
        for ncp in self.ncp_layers:
            h, _ = ncp(h)
        return torch.sigmoid(self.head(h).squeeze(-1))
