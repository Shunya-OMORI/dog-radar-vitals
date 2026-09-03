"""レーダ波形からR波heatmapを推定する1D Attention U-Net。

`rpeak_unet1d.py`（プレーンなU-Net、スキップ接続をそのままconcatする）と
バックボーンは完全に同じで、**スキップ接続にAttention Gateを挟む点だけ**が異なる
(単一変数の比較用)。

対応する先行研究: Oktay ら, "Attention U-Net: Learning Where to Look for the Pancreas,"
MIDL 2018 (arXiv:1804.03999)。デコーダ側の粗い特徴(ゲート信号g)を使って、
エンコーダ側のスキップ特徴(x)のどのサンプルを通すかを学習する。R波heatmapは
4秒窓のうち9割以上が背景(§3.7と同じ不均衡)なので、「どこを見るか」を明示的に
絞るこの機構が、素のU-Netより効くかを検証する動機がある。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dog_radar_vitals.models.deep.ecg_unet1d import ConvBlock


class AttentionGate1D(nn.Module):
    """Oktay ら Fig.2 の1次元版。g(デコーダ側の粗い特徴)でx(エンコーダ側のスキップ特徴)を
    ゲーティングする。alpha in [0,1] をxに掛けて、通す量をサンプルごとに絞る。
    """

    def __init__(self, x_channels: int, g_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta_x = nn.Conv1d(x_channels, inter_channels, kernel_size=1)
        self.phi_g = nn.Conv1d(g_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv1d(inter_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x: (B, x_channels, T_x)  g: (B, g_channels, T_g)。T_gをT_xに合わせてから使う。"""
        if g.shape[-1] != x.shape[-1]:
            g = F.interpolate(g, size=x.shape[-1], mode="linear", align_corners=False)
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = torch.relu(theta_x + phi_g)
        alpha = torch.sigmoid(self.psi(f))  # (B, 1, T_x)、サンプルごとの通過率
        return x * alpha


class RPeakAttentionUNet1D(nn.Module):
    def __init__(
        self, in_channels: int = 50, base_channels: int = 32, depth: int = 3, kernel_size: int = 9,
    ) -> None:
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
        self.gates = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for _ in range(depth):
            self.ups.append(nn.ConvTranspose1d(ch, in_ch, kernel_size=4, stride=2, padding=1))
            # gate: x_channels=in_ch(スキップ側), g_channels=in_ch(アップ後のデコーダ側), inter=in_ch//2
            self.gates.append(AttentionGate1D(x_channels=in_ch, g_channels=in_ch, inter_channels=max(1, in_ch // 2)))
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

        for up, gate, dec_block, skip in zip(self.ups, self.gates, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            gated_skip = gate(skip, h)  # ← ここがプレーンなU-Netとの唯一の差分
            h = dec_block(torch.cat([h, gated_skip], dim=1))

        return torch.sigmoid(self.head(h).squeeze(1))
