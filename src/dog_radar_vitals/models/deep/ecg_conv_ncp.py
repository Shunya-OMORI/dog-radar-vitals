"""NCP進化系: 局所畳み込み前処理 + CfC(mixed_memory対応)のハイブリッド（本命1の拡張版）。

201-224フェーズで、`ecg_conformer.py`（局所畳み込み+Self-Attention）がTransformer単体の
未収束問題を解消したのと同じ発想を、NCPに適用する。生のRCG(50ch)を直接CfCへ渡すのではなく、
QRS幅程度の受容野を持つ局所dilated convで前処理してから渡すことで、(a) CfCの各タイムステップに
より情報量の多い局所特徴を与え、(b) ノイズ頑健性実験で判明したNCPの弱点（生信号のノイズに
敏感）を、畳み込みの平滑化・パターン抽出で補うことを狙う。

さらに`mixed_memory=True`（LSTM様のメモリセルでCfCを補強、長期依存の学習を助ける、
`ncps`ライブラリのオプション）を有効にし、複数層のCfCを積み重ねられるようにする
（単層の`ecg_ncp.py`に対する拡張）。
"""
from __future__ import annotations

import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn

from dog_radar_vitals.models.deep.ecg_conformer import ConvSubsampler


class ECGConvNCP(nn.Module):
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
        """x: (batch, seq_len, in_channels) -> (batch, seq_len)"""
        h = self.local_conv(x)  # (batch, seq_len, conv_channels)
        for ncp in self.ncp_layers:
            h, _ = ncp(h)
        return self.head(h).squeeze(-1)
