"""学習可能な時間遅延τ補正モジュール(2026-07-30)。

radarODE論文のODEデコーダは、「機械的活動(レーダで見える動き)は電気的活動(ECG)より
短い時間τだけ遅れる」という生理学的知見に基づき、生成したECG波形を時間τだけシフトする
学習可能な補正を組み込んでいる("the solution of the ODEs will be shifted to the left with
time τ")。論文はτの具体的な予測・適用方法までは記載していない。

診断実験(248の予測、`reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)で、
±20サンプル(200サンプル中)のシフト探索を許すと相関が0.36→0.54まで跳ね上がることを確認した
(ただしシフト方向は左右対称で、系統的な生理学的遅延ではなく拍境界推定のノイズによると推測)。
このタイミングズレを補正するため、Spatial Transformer Networks(Jaderberg et al. 2015)の
1次元版である"Temporal Transformer" layer(localization networkが変形パラメータを予測し、
微分可能なリサンプラーで適用する設計、時系列アライメント文献で提案されている手法)に着想した
`TemporalShiftHead`を実装する。localization networkは入力特徴(ECG自体ではなくRCG/SST由来の
中間特徴)からスカラーの時間シフトτを予測するため、推論時(正解ECGが無い状況)でも使える。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def shift_signal(x: torch.Tensor, tau_frac: torch.Tensor) -> torch.Tensor:
    """1次元信号(N, T)を、系列長に対する割合`tau_frac`(N,)だけ微分可能にシフトする。

    `torch.nn.functional.grid_sample`を使い、正の`tau_frac`で信号を時間軸上「前方」に
    シフトする(出力[i] = 入力[i + tau_frac*T]の線形補間)。シフトで生じる境界の空白は
    端の値を複製する(`padding_mode="border"`)。
    """
    n, t = x.shape
    x_ = x.view(n, 1, 1, t)
    base_grid = torch.linspace(-1, 1, t, device=x.device, dtype=x.dtype).view(1, 1, t, 1).expand(n, 1, t, 1)
    shift = (2.0 * tau_frac).view(n, 1, 1, 1)
    grid_x = base_grid + shift
    grid_y = torch.zeros_like(grid_x)
    grid = torch.cat([grid_x, grid_y], dim=-1)
    shifted = F.grid_sample(x_, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return shifted.view(n, t)


def shift_batch_per_example(x: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """x: (N, T)を、サンプルごとに異なる整数シフト`shifts`(N,)だけシフトする(端は複製)。
    `torch.gather`によるインデックス選択なので`x`に関して微分可能。
    """
    n, t = x.shape
    base = torch.arange(t, device=x.device).unsqueeze(0).expand(n, t)
    idx = (base - shifts.unsqueeze(1)).clamp(0, t - 1)
    return torch.gather(x, 1, idx)


def oracle_best_shift(pred: torch.Tensor, true: torch.Tensor, max_shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """正解`true`を知っている前提で、`pred`をサンプルごとに整数シフトして最も相関が高くなる
    シフト量(N,)とその相関(N,)を全探索で求める(oracle、学習時のshift-invariant訓練にのみ
    使用可、推論時には使えない)。2026-07-30の診断で257(タイトクロップCNN)のtest予測に
    適用したところ、相関0.203→0.586まで跳ね上がることを確認した(詳細は
    `reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。
    """
    n = pred.shape[0]
    best_corr = torch.full((n,), -2.0, device=pred.device)
    best_shift = torch.zeros(n, dtype=torch.long, device=pred.device)
    for shift in range(-max_shift, max_shift + 1):
        shifted = shift_batch_per_example(pred, torch.full((n,), shift, device=pred.device, dtype=torch.long))
        pc = shifted - shifted.mean(dim=1, keepdim=True)
        tc = true - true.mean(dim=1, keepdim=True)
        corr = (pc * tc).sum(dim=1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)
        improve = corr > best_corr
        best_corr = torch.where(improve, corr, best_corr)
        best_shift = torch.where(improve, torch.full_like(best_shift, shift), best_shift)
    return best_shift, best_corr


class TemporalShiftHead(nn.Module):
    """中間特徴(N, C, T_feat)からスカラーの時間シフトτ(系列長に対する割合)を予測し、
    与えられた波形(N, T_out)へ適用する。Temporal Transformer layer(1D版Spatial Transformer)
    に着想。`max_shift_frac`でτの範囲を[-max_shift_frac, max_shift_frac]に制限する
    (診断実験の結果から、既定0.15=200サンプル中最大30サンプル相当)。

    v2(2026-07-30): 初版は`AdaptiveAvgPool1d(1)`で時間軸を1点に潰してからτを予測しており、
    「このビートが早いか遅いか」を判断するのに必要な位置情報(時間軸上のどこに特徴がある
    か)を破壊していた。実際、この設計でτは学習後もほぼ0付近(平均-0.45サンプル、標準偏差
    0.09サンプル/200)に留まり、機能していなかった(詳細は
    `reports/mmecg_comparison/prior_work_accuracy_comparison.md`参照)。v2は
    `AdaptiveAvgPool1d(pooled_time)`で時間軸を粗く残し(既定8点)、flattenしてから
    線形層に通すことで、大まかな位置情報を保ったままτを予測できるようにした。
    """

    def __init__(self, in_channels: int, hidden: int = 16, max_shift_frac: float = 0.15, pooled_time: int = 8) -> None:
        super().__init__()
        self.max_shift_frac = max_shift_frac
        self.loc = nn.Sequential(
            nn.AdaptiveAvgPool1d(pooled_time),
            nn.Flatten(),
            nn.Linear(in_channels * pooled_time, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )
        # tanh出力の初期値は乱数に依存し得るが、学習初期はshift=0付近から始める方が安定するため
        # 最終層をゼロ初期化する(shift=0=補正なしから学習を始める)。
        nn.init.zeros_(self.loc[-2].weight)
        nn.init.zeros_(self.loc[-2].bias)

    def predict_tau(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, C, T_feat) -> tau_frac: (N,)。"""
        return self.loc(feat).squeeze(-1) * self.max_shift_frac

    def forward(self, feat: torch.Tensor, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau = self.predict_tau(feat)
        return shift_signal(signal, tau), tau
