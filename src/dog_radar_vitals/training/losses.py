"""heatmapキーポイント検出向けの損失関数。

背景（2026-08-06、ユーザ指摘）: 極大点の後処理（`data/rpeaks.py`の各`extract_peaks_*`）は
既に出力されたheatmapの中から候補を選ぶことしかできず、モデルがそもそも真のR波位置で
heatmap値を十分上げられていなければ（見逃し=false negative）後処理では取り返せない。
したがって、heatmap生成モデルの学習は素朴なBCE（適合率・再現率を対称に扱う）ではなく、
見逃しを抑える方向に非対称な損失であるべきという設計変更。
"""
from __future__ import annotations

import torch


def heatmap_focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """CornerNet/CenterNetのpenalty-reduced focal loss（keypoint heatmap用）。

    対応する先行研究: Law & Deng, "CornerNet: Detecting Objects as Paired Keypoints",
    ECCV 2018 / Zhou et al., "Objects as Points" (CenterNet), 2019。

    通常のBCEは全画素を対称に扱うため、正例（R波中心）1点に対し負例（背景）が
    圧倒的に多い今回のheatmapでは、モデルは「全体を薄く光らせる」ことでも損失を
    下げられてしまい、真のピーク位置の値を十分高くする動機が弱い。focal lossは
    (1-pred)^alphaで「まだ低い正例」への勾配を強く保ち、pred^alphaで「既に低い負例」
    への勾配を弱め、(1-target)^betaでガウシアンの裾（真のR波のすぐ近く）への
    誤検出ペナルティを緩めることで、正例（見逃し防止）に学習を集中させる。
    """
    pred = pred.clamp(eps, 1 - eps)
    pos_mask = (target >= 1.0 - eps).float()
    neg_mask = 1.0 - pos_mask

    pos_loss = -torch.log(pred) * (1 - pred).pow(alpha) * pos_mask
    neg_loss = -torch.log(1 - pred) * pred.pow(alpha) * (1 - target).pow(beta) * neg_mask

    n_pos = pos_mask.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos
