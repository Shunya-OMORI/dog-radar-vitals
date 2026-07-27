"""レーダ波形からECG波形を推定するNCP(Neural Circuit Policy)モデル（本命の1つ）。

`ncps.torch.CfC`（Closed-form Continuous-time RNN）を`ncps.wirings.AutoNCP`で
疎結合配線し、通常の全結合LSTM/Transformerより少ないパラメータ数で時間的因果構造を
学習できるかを検証する。`CfC(input_size, units=wiring, batch_first=True)`は
`(batch, seq_len, wiring.output_dim)`を返す（`units`にWiringを渡すと自動的に
`wired_mode`になる。ncpsのAPI仕様はncps==1.0.1で確認済み）。AutoNCPの出力ニューロン数
(motor neurons)をそのまま最終出力次元として使い、通常の全結合readout層は追加しない
（配線構造自体をreadoutとして使うのがNCPの流儀のため）。
"""
from __future__ import annotations

import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn


class ECGWaveformNCP(nn.Module):
    def __init__(self, in_channels: int = 2, units: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        wiring = AutoNCP(units, output_dim)
        self.rnn = CfC(in_channels, wiring, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
        h, _ = self.rnn(x)
        return h.squeeze(-1)
