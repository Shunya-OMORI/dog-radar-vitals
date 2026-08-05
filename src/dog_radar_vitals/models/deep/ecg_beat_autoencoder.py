"""単一拍ECG波形のオートエンコーダ。ユーザ提案の「ECG基盤モデル/事前学習モデル」路線。

McSharry型ODEデコーダ（`radarode_sceg.py`）は物理モデルに基づく固定的な形状事前分布だが、
こちらは実際の臨床ECG（Schellenberger、レーダ非依存）から**データ駆動で**PQRST波形の形を
学習する。学習済み(凍結)のdecoderを`PretrainedShapeDecoder`（`radarode_sceg.py`に追加）が
再利用し、SCEGは「レーダ特徴からこの学習済み形状空間への写像」だけを学習すればよくなる、
という設計（詳細は`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照）。
"""
from __future__ import annotations

import torch
from torch import nn


class BeatEncoder(nn.Module):
    """単一拍ECG(N, T) -> 潜在ベクトル(N, latent_dim)。"""

    def __init__(self, in_len: int = 200, latent_dim: int = 16, channels: tuple[int, ...] = (16, 32, 64)) -> None:
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        reduced_len = in_len // (2 ** len(channels))
        self.proj = nn.Linear(channels[-1] * reduced_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.unsqueeze(1))
        return self.proj(h.flatten(1))


class BeatDecoder(nn.Module):
    """潜在ベクトル(N, latent_dim) -> 単一拍ECG(N, out_len)。"""

    def __init__(self, out_len: int = 200, latent_dim: int = 16, channels: tuple[int, ...] = (64, 32, 16)) -> None:
        super().__init__()
        self.out_len = out_len
        self.init_len = out_len // (2 ** len(channels))
        self.init_ch = channels[0]
        self.proj = nn.Linear(latent_dim, channels[0] * self.init_len)

        layers = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        layers.append(nn.ConvTranspose1d(in_ch, 1, kernel_size=4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).reshape(z.shape[0], self.init_ch, self.init_len)
        out = self.deconv(h).squeeze(1)
        if out.shape[-1] != self.out_len:
            out = nn.functional.interpolate(out.unsqueeze(1), size=self.out_len, mode="linear", align_corners=False).squeeze(1)
        return out


class BeatAutoencoder(nn.Module):
    def __init__(self, seq_len: int = 200, latent_dim: int = 16) -> None:
        super().__init__()
        self.encoder = BeatEncoder(seq_len, latent_dim)
        self.decoder = BeatDecoder(seq_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
