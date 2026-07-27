"""レーダ波形からECG波形を推定するConformer型モデル（局所畳み込み+大域Self-Attention）。

`ecg_transformer.py`（素のTransformer Encoder）は201-215フェーズでrpeak版が未収束（F1=0）
だった。仮説: Self-Attentionは平行移動不変性を持たず、QRS複合波のような鋭く局所的な形状を
少ないtrain被験者数(7名)から学習するのが難しい。局所畳み込み(translation-equivariant、
QRS幅に近い受容野)を前段に置き、局所的な形状検出という帰納バイアスを明示的に与えたうえで
Self-Attention（拍と拍の間の周期的な長距離依存を捉える）に渡すConformer型構成
（Gulati et al. 2020, 音声認識で提案された局所畳み込み+Self-Attentionの複合アーキテクチャと
同じ設計思想）にすることで、この収束の悪さを解消できるという仮説を検証する。
"""
from __future__ import annotations

import torch
from torch import nn

from dog_radar_vitals.models.deep.transformer import SinusoidalPositionalEncoding


class ConvSubsampler(nn.Module):
    """QRS幅程度の受容野を持つ局所dilated conv。翻訳不変な局所特徴抽出を担う（解像度は変えない）。"""

    def __init__(self, in_channels: int, d_model: int, kernel_size: int = 15, n_layers: int = 3) -> None:
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_layers):
            dilation = 2**i
            layers += [
                nn.Conv1d(in_ch, d_model, kernel_size, padding="same", dilation=dilation),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ]
            in_ch = d_model
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len, d_model)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class ECGWaveformConformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 50,
        d_model: int = 128,
        conv_kernel_size: int = 15,
        conv_layers: int = 3,
        n_heads: int = 4,
        n_transformer_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.local_conv = ConvSubsampler(in_channels, d_model, conv_kernel_size, conv_layers)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
        h = self.local_conv(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)
