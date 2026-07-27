"""PQRSTグラフ(5ノード×2特徴)が「本物らしいか」を判定する小型MLP判別器（GAN拡張用）。

`beatgraph_gnn.py`の生成器が出す予測グラフと、データセットの真のグラフを見分けることで、
MSE回帰だけでは学習しにくい「PQRST波形として尤もらしい形」を正則化として与える狙い
（条件なし判別器: レーダ入力には依存せず、グラフの形の妥当性のみを見る）。
"""
from __future__ import annotations

import torch
from torch import nn


class BeatGraphDiscriminator(nn.Module):
    def __init__(self, n_nodes: int = 5, n_node_features: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes * n_node_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph: torch.Tensor) -> torch.Tensor:
        """graph: (batch, n_nodes, n_node_features) -> (batch,) real/fakeロジット"""
        return self.net(graph.flatten(start_dim=1)).squeeze(-1)
