"""τ(時間遅延補正)専用のスカラー回帰モデル。family=`mmecg_tau_predictor`。

256/258/261/262(いずれも波形回帰用の`ECGWaveformCNN1D`の中間特徴、または周波数軸を
平均poolingで潰した特徴からτを予測)では、oracle疑似ラベルτへの教師あり回帰損失を
足しても予測τがオラクルτとほぼ無相関のまま(261: -0.0001、262: -0.0038)だった。
これは「波形回帰に特化したアーキテクチャ(周波数軸を早々に潰す設計)がτ推定に
そもそも向いていない」可能性を検証していなかったための混同かもしれない。

`TauOnlyCNN`は波形回帰を一切行わず、SST(N, 50, F, T)全体を周波数軸を潰さずに
2D Conv2dで直接処理し(`radarode_sceg.py`の`Backbone`と同系統だが軽量)、
スカラーτの回帰のみを目的とする専用アーキテクチャ。タスクが違えば適したアーキテクチャも
違うかもしれない、というユーザ指摘に基づく設計。
"""
from __future__ import annotations

import torch
from torch import nn


class _Conv2dDownsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TauOnlyCNN(nn.Module):
    """SST(N, 50, F, T) -> スカラーτ_frac(N,)。波形は一切予測しない、τ専用回帰モデル。"""

    def __init__(
        self,
        in_channels: int = 50,
        channels: tuple[int, ...] = (64, 128, 256),
        hidden: int = 64,
        dropout: float = 0.1,
        max_shift_frac: float = 0.15,
    ) -> None:
        super().__init__()
        self.max_shift_frac = max_shift_frac
        blocks = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(_Conv2dDownsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        # ゼロ初期化: 学習開始時はτ=0(補正なし)から始める(TemporalShiftHeadと同じ理由)。
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, sst: torch.Tensor) -> torch.Tensor:
        h = self.blocks(sst)
        h = self.pool(h)
        return self.head(h).squeeze(-1) * self.max_shift_frac
