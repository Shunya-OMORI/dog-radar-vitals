"""レーダ波形からR波heatmapを推定する1D U-Net。`ecg_unet1d.py`と入力・バックボーンは同じだが、
出力層にsigmoidを通す点が異なる。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dog_radar_vitals.models.deep.ecg_unet1d import ConvBlock


class RPeakUNet1D(nn.Module):
    def __init__(self, in_channels: int = 50, base_channels: int = 32, depth: int = 3, kernel_size: int = 9) -> None:
        super().__init__()
        self.depth = depth

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_channels
        in_ch = in_channels
        for _ in range(depth):
            self.enc_blocks.append(ConvBlock(in_ch, ch, kernel_size))
            self.downs.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))
            in_ch = ch
            ch *= 2

        self.bottleneck = ConvBlock(in_ch, ch, kernel_size)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose1d(ch, in_ch, kernel_size=4, stride=2, padding=1))
            self.dec_blocks.append(ConvBlock(in_ch * 2, in_ch, kernel_size))
            ch = in_ch
            in_ch //= 2

        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)、値域[0,1]のheatmap。"""
        h = x.transpose(1, 2)

        skips = []
        for enc_block, down in zip(self.enc_blocks, self.downs):
            h = enc_block(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec_block, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = dec_block(torch.cat([h, skip], dim=1))

        return torch.sigmoid(self.head(h).squeeze(1))
