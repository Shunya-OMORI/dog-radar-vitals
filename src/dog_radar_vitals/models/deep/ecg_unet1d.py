"""レーダ波形からECG波形を推定する1D U-Net（多重解像度エンコーダ・デコーダ）。

`ecg_cnn1d.py`（dilated convを1本道で積み、時間解像度を落とさず全層処理する構成）に対し、
本モデルはstride-2畳み込みで段階的にダウンサンプリングし、対応する転置畳み込みで
アップサンプリングして戻すU-Net構成にする。設計の数学的根拠（LifWavNet, arXiv:2510.27692の
主張と同じ発想）: ECGの鋭いQRS複合波（高周波成分、~50-80ms幅）は微細な時間分解能を持つ
浅いエンコーダ層とスキップ接続で捉え、緩やかなP/T波（低周波成分）はダウンサンプリングされた
ボトルネックで捉える、という周波数帯ごとの分離を明示的な帰納バイアスとして与える。
1本道のdilated convスタックはこの分離を持たない。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGWaveformUNet1D(nn.Module):
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
            self.dec_blocks.append(ConvBlock(in_ch * 2, in_ch, kernel_size))  # skip接続で2倍
            ch = in_ch
            in_ch //= 2

        self.head = nn.Conv1d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
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
                # seq_lenが2^depthの倍数でない場合、ストライド2畳み込みの丸めでスキップ接続と
                # 1サンプルずれることがある。線形補間でskip側の長さに合わせる（実際に使う
                # window_sec=4,fs=200Hz=800サンプルは2^3で割り切れるため通常は発生しない）。
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = dec_block(torch.cat([h, skip], dim=1))

        return self.head(h).squeeze(1)
