"""posXYZ(実座標)からk近傍グラフを構築する共有ユーティリティ。

`ecg_spatial_fusion.py`は50点全結合のSelf-Attentionで空間融合していたが、これは
「どの点も等しく参照しあう」近似であり、実際の3D距離構造を使っていない。MMPoint-GNN
(Gong et al. 2021、mmWave point cloudのHAR向けGNN)等の先行研究に倣い、**実際の3D距離に
基づくk近傍グラフ**でGATConvを使うことで、空間的に近い点同士の情報融合という、より
物理的に妥当な帰納バイアスを与える（`torch_geometric.nn.knn_graph`は`pyg-lib`を要求し
本環境では未導入のため、素朴なtorch実装で代替する）。
"""
from __future__ import annotations

import torch


def knn_edge_index(pos: torch.Tensor, k: int) -> torch.Tensor:
    """pos: (n_points, 3) -> edge_index (2, n_points*k)。自分自身を除く距離最小k点への有向エッジ。"""
    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)  # (n_points, n_points)
    dist.fill_diagonal_(float("inf"))
    _, nn_idx = torch.topk(dist, k, dim=1, largest=False)  # (n_points, k)

    n_points = pos.shape[0]
    src = torch.arange(n_points, device=pos.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nn_idx.reshape(-1)
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])  # 無向化


def batched_knn_edge_index(pos: torch.Tensor, k: int, n_repeat: int = 1) -> torch.Tensor:
    """pos: (batch, n_points, 3) -> 各サンプルのk近傍グラフをバッチ結合したedge_index。

    `n_repeat`は、同一サンプルのグラフ構造をタイムステップ方向に使い回す際の複製回数
    （posXYZは録音内で時間変化しないため、T'個の縮約タイムステップすべてで同じグラフを使う）。
    ノードの並び順は `(batch, n_repeat, n_points)` をflattenした順（= batch*n_repeat個の
    グラフが連続して並ぶ）を前提とする。
    """
    batch_size, n_points, _ = pos.shape
    edge_indices = []
    for b in range(batch_size):
        base_edges = knn_edge_index(pos[b], k)
        for r in range(n_repeat):
            offset = (b * n_repeat + r) * n_points
            edge_indices.append(base_edges + offset)
    return torch.cat(edge_indices, dim=1)
