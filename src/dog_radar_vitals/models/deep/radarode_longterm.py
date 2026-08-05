"""radarODEの長期ECG再構成ネットワーク(Table III・Fig.2c)。family=`mmecg_radarode_longterm`。

生のRCG時系列(50ch, 4秒窓)をResNet型downsampling encoderで時間特徴に変換し、
`RadarODESCEG`が1心拍ごとに生成した単一拍ECG片を時系列順に連結した
「morphological reference」(4秒窓分の連続信号)と2チャネルにスタックしたうえで、
**非自己回帰の** dilated Conv1d(TCN、9層・dilation 2^i)で融合し最終ECGを出力する。

`chen2022_reconstructor.py`（Chen et al.の式(13)通りの真の自己回帰TCNデコーダ、
teacher forcing下の見かけの精度とexposure biasによる崩壊を実証済み）とは異なり、
このTCNは1回のforward passで全時刻を同時に出力する（自己回帰生成を一切行わない）。
これがradarODEの長期再構成が我々のChen et al.再現より頑健である鍵だと考えられる
（詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照）。
"""
from __future__ import annotations

import torch
from torch import nn


class _ResidualDownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class LongTermEncoder(nn.Module):
    def __init__(self, in_channels: int = 50, channels: tuple[int, ...] = (128, 256, 512)) -> None:
        super().__init__()
        blocks = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(_ResidualDownsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class LongTermDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: tuple[int, ...] = (128, 16, 1)) -> None:
        super().__init__()
        blocks = []
        in_ch = in_channels
        for out_ch in out_channels:
            blocks.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if out_ch != out_channels[-1]:
                blocks += [nn.BatchNorm1d(out_ch), nn.ReLU()]
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _DilatedConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2  # 非自己回帰: 因果性を課さず両側にpadding
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.conv(x))


class NonAutoregressiveTCNFusion(nn.Module):
    """2ch(temporal feature, morphological reference)を非自己回帰の9層dilated Conv1dで融合する。"""

    def __init__(self, hidden: int = 16, n_layers: int = 9) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(2, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([_DilatedConvBlock(hidden, dilation=2**i) for i in range(n_layers)])
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, temporal: torch.Tensor, morphological_ref: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(torch.cat([temporal, morphological_ref], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.head(x).squeeze(1)


class RadarODELongTerm(nn.Module):
    def __init__(self, n_points: int = 50, channels: tuple[int, ...] = (128, 256, 512), tcn_hidden: int = 16) -> None:
        super().__init__()
        self.encoder = LongTermEncoder(n_points, channels)
        self.decoder = LongTermDecoder(channels[-1])
        self.fusion = NonAutoregressiveTCNFusion(tcn_hidden)

    def forward(self, rcg: torch.Tensor, morphological_ref: torch.Tensor) -> torch.Tensor:
        """rcg: (batch, 50, T)。morphological_ref: (batch, T)（SCEGが生成した参照波形）。-> (batch, T)。"""
        h = self.encoder(rcg)
        temporal = self.decoder(h)
        if temporal.shape[-1] != rcg.shape[-1]:
            temporal = nn.functional.interpolate(temporal, size=rcg.shape[-1], mode="linear", align_corners=False)
        return self.fusion(temporal, morphological_ref.unsqueeze(1))
