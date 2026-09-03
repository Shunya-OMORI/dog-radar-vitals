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


def heatmap_bce_localexp_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loc_weight: float = 0.1,
    half_win: int = 17,
    eps: float = 1e-6,
) -> torch.Tensor:
    """BCE + 局所soft-argmax位置損失（heatmap重心復号との学習/推論整合）。

    背景 (2026-08-07): 推論側にサブサンプル重心復号(`refine_peaks_centroid`, ±85ms窓)を
    導入してRR-MAE 11.76→8.41msを達成したが、残る誤差の主体はピークごとの位置ジッタ
    (matched timing error std ~27ms)で、BCEはピクセル単位の分類損失のため「山の質量中心が
    正解位置に一致すること」を直接は罰しない。本損失は各正解R波中心の±half_winサンプル
    窓内で、復号と同じ「窓内最小値を引いた重み付き重心」のオフセット期待値を計算し、
    そのゼロからのずれをL1で罰する（＝復号器が読む統計量そのものを教師位置に合わせる）。

    対応する先行研究: Sun et al., "Integral Human Pose Regression" (ECCV 2018) /
    Nibali et al., "Numerical Coordinate Regression with Convolutional Neural
    Networks" (DSNT, 2018)。heatmapのsoft-argmax(期待値)を微分可能な座標として
    直接座標損失をかける枠組み。本実装はマルチピークの1D版として、各GT中心の
    局所窓に限定して期待値を取る。

    Parameters
    ----------
    loc_weight: 位置損失の係数。位置項は「サンプル単位のオフセット絶対値の平均」
        (1サンプル=5ms@200Hz)。
    half_win: 窓の半幅[サンプル]。既定17≒85ms@200Hz(復号側と一致させる)。
    """
    pred_c = pred.clamp(eps, 1 - eps)
    bce = torch.nn.functional.binary_cross_entropy(pred_c, target)

    centers = (target >= 1.0 - 1e-4).nonzero(as_tuple=False)  # (N, 2): [batch, t]
    if centers.numel() == 0:
        return bce
    t_idx = centers[:, 1]
    valid = (t_idx >= half_win) & (t_idx < pred.shape[1] - half_win)
    if valid.sum() == 0:
        return bce
    b_idx, t_idx = centers[valid, 0], t_idx[valid]

    offsets = torch.arange(-half_win, half_win + 1, device=pred.device)
    win = pred[b_idx.unsqueeze(1), t_idx.unsqueeze(1) + offsets.unsqueeze(0)]  # (N, 2*hw+1)
    w = win - win.min(dim=1, keepdim=True).values
    exp_offset = (w * offsets.float()).sum(dim=1) / (w.sum(dim=1) + eps)
    loc = exp_offset.abs().mean()
    return bce + loc_weight * loc


def heatmap_weighted_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 10.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """正例側に重みを掛けた BCE。見逃し(recall不足)を直接罰する。

    背景 (2026-08-25): 11-fold LOSO の被験者別診断で、precision は 0.93〜1.00 と高いのに
    recall が 0.02〜0.99 と大きく落ちる被験者が複数あった。つまりモデルは「拍でない」と
    答える方に偏っている。教師 heatmap は sigma=15 ms のガウシアンなので、4 秒窓
    (800サンプル) のうち実質的に正例と言える領域は 5 拍 × 数十サンプルに過ぎず、
    残り 9 割以上が 0 である。素の BCE はこの多数派に引きずられる。

    追跡したいのは RR 間隔とその変動 (HRV) であり、**拍を 1 つ見逃すとその区間の
    RR 間隔が倍になって RMSSD を壊す**。誤検出より見逃しの方が HRV への害が大きい。
    したがって正例側を重く見る。

    対応する先行研究: クラス不均衡に対する重み付き交差エントロピーは標準的な扱いで、
    heatmap 型キーポイント検出では CornerNet (Law & Deng, ECCV 2018) の focal 変種が
    よく使われる (`heatmap_focal_loss` として実装済み)。focal が「難しい例に集中する」
    のに対し、こちらは「正例そのものを重く見る」ので、狙いが違う。

    Parameters
    ----------
    pos_weight: 正例側の重み。1.0 で通常の BCE と一致する。
    """
    pred = pred.clamp(eps, 1.0 - eps)
    loss = -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    return loss.mean()


def adaptive_wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    omega: float = 14.0,
    theta: float = 0.5,
    epsilon: float = 1.0,
    alpha: float = 2.1,
) -> torch.Tensor:
    """Adaptive Wing Loss（Wang ら, ICCV 2019, arXiv:1904.07399）。

    背景 (2026-08-28、ユーザ指示による先行研究サーベイ): 顔ランドマークのheatmap回帰でも
    前景1.2%程度という、本タスクのR波(σ=15msガウシアンで4秒窓の約6-7%)と同程度の
    強いクラス不均衡が問題になっており、この論文はBCE系ではなくL1に近い罰則関数の
    形自体を変えることで対処している。target値(target^alpha近傍)に応じて指数alphaが
    連続的に変わり、正解に近い(target≈1)画素では急峻(誤差に敏感)、遠い(target≈0)画素では
    ゆるやか(外れ値に鈍感)になる。本実装は原論文Eq.(4)(5)通り。

    heatmap_focal_loss/heatmap_weighted_bce_lossがBCE系(交差エントロピー)であるのに対し、
    本損失は絶対誤差ベース(L1/Wing系)であり、損失関数の"族"自体が異なる比較対象になる。
    """
    delta = (pred - target).abs()
    a = alpha - target  # target=1(正解ちょうど)でalpha-1に近づき、target=0でalphaに近づく
    pow_a = (theta / epsilon) ** a
    pow_a1 = (theta / epsilon) ** (a - 1.0)

    A = omega * (1.0 / (1.0 + pow_a)) * a * pow_a1 / epsilon
    C = theta * A - omega * torch.log(1.0 + pow_a)

    is_small = delta < theta
    loss_small = omega * torch.log(1.0 + (delta / epsilon).pow(a))
    loss_large = A * delta - C
    return torch.where(is_small, loss_small, loss_large).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice損失（QRS検出のU-Net系文献で矩形マスク教師と併用される定番、例: arXiv:1912.09223）。

    BCEはサンプルごとに独立に評価するため不均衡に弱いが、Diceは予測領域と正解領域の
    重なり(集合としてのIoUに近い量)を直接最大化するため、疎な正例に対しても勾配が
    消えにくい。`build_peak_box`(矩形マスク教師)とセットで使う想定。
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def box_bce_dice_loss(pred: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    """矩形マスク教師(build_peak_box)用: BCE + Dice loss(U-Net系QRS検出文献の定番構成)。"""
    bce = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)
    return bce + dice_weight * dice_loss(pred, target)
