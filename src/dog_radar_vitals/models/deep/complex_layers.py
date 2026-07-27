"""複素領域モデル用の共有レイヤ。

PyTorch(2.5系)の`nn.Conv1d`は`dtype=torch.complex64`を指定するとそのまま複素畳み込みとして
動作する（複素の重みで実装すれば自動的に複素乗算になる。バックワードも動作確認済み）ため、
複素畳み込み自体は自作しない。一方`nn.BatchNorm1d`は複素dtypeで`NotImplementedError`になる
ため、v1(`ecg_complex_cnn.py`)では正規化層を使わず`ModReLU`のみで構成していた。
v2(`ecg_complex_cnn_v2.py`)では`ComplexBatchNorm1d`を追加し、深い複素ネットワークの学習を
安定化させる（Trabelsi et al. 2018, *Deep Complex Networks*が「素朴に実部・虚部を独立に
実数BNする方式は6実験中5つがNaNで発散したが、本実装(共分散行列で白色化する方式)は
6実験とも収束した」と報告している設計）。
"""
from __future__ import annotations

import torch
from torch import nn


class ModReLU(nn.Module):
    """z -> ReLU(|z| + b) * (z / |z|)。振幅にバイアス付きReLUをかけ、位相は保存する。"""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_features, seq_len) complex64"""
        magnitude = x.abs()
        biased_magnitude = magnitude + self.bias.view(1, -1, 1)
        scale = torch.relu(biased_magnitude) / (magnitude + 1e-6)
        return x * scale.to(x.dtype)


class ComplexBatchNorm1d(nn.Module):
    """複素版バッチ正規化（Trabelsi et al. 2018の定式化）。

    実数BNは「スカラーの分散で割る」ことで正規化するが、複素数は(実部,虚部)の2次元ベクトルの
    ため、素朴に実部・虚部それぞれをスカラー分散で正規化すると、実部と虚部の間の相関
    （位相情報の一部）を無視してしまい学習が不安定になる（原論文で報告されている発散の原因）。
    本実装は、チャネルごとに(実部,虚部)の2×2共分散行列

        V = [[Vrr, Vri], [Vri, Vii]]

    を求め、その逆平方根W = V^{-1/2}を(実部,虚部)ベクトルに左から掛けて白色化する
    （白色化後の共分散が単位行列に近づく）。2×2対称行列の逆平方根は閉形式で計算できる:

        s = sqrt(det V) = sqrt(Vrr*Vii - Vri^2)
        t = sqrt(tr V + 2s) = sqrt(Vrr + Vii + 2s)
        W = (1/(s*t)) * [[Vii + s, -Vri], [-Vri, Vrr + s]]

    白色化後、学習可能な2×2半正定値スケール行列γ（3自由度: γrr, γri, γii）とバイアスβ
    (実部・虚部)を適用する。学習時はバッチ統計、評価時は移動平均統計（実数BNと同じ運用）を使う。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma_rr = nn.Parameter(torch.full((num_features,), 1.0 / (2**0.5)))
        self.gamma_ii = nn.Parameter(torch.full((num_features,), 1.0 / (2**0.5)))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_vrr", torch.full((num_features,), 1.0 / (2**0.5)))
        self.register_buffer("running_vii", torch.full((num_features,), 1.0 / (2**0.5)))
        self.register_buffer("running_vri", torch.zeros(num_features))
        self.register_buffer("running_mean_r", torch.zeros(num_features))
        self.register_buffer("running_mean_i", torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_features, seq_len) complex64"""
        xr, xi = x.real, x.imag

        if self.training:
            # チャネルごとにbatch*seq_len次元で統計を取る（実数BatchNorm1dと同じ縮約軸）。
            mean_r = xr.mean(dim=(0, 2))
            mean_i = xi.mean(dim=(0, 2))
            cr = xr - mean_r.view(1, -1, 1)
            ci = xi - mean_i.view(1, -1, 1)
            vrr = (cr * cr).mean(dim=(0, 2)) + self.eps
            vii = (ci * ci).mean(dim=(0, 2)) + self.eps
            vri = (cr * ci).mean(dim=(0, 2))

            with torch.no_grad():
                self.running_mean_r.mul_(1 - self.momentum).add_(self.momentum * mean_r)
                self.running_mean_i.mul_(1 - self.momentum).add_(self.momentum * mean_i)
                self.running_vrr.mul_(1 - self.momentum).add_(self.momentum * vrr)
                self.running_vii.mul_(1 - self.momentum).add_(self.momentum * vii)
                self.running_vri.mul_(1 - self.momentum).add_(self.momentum * vri)
        else:
            mean_r, mean_i = self.running_mean_r, self.running_mean_i
            vrr, vii, vri = self.running_vrr, self.running_vii, self.running_vri
            cr = xr - mean_r.view(1, -1, 1)
            ci = xi - mean_i.view(1, -1, 1)

        delta = vrr * vii - vri * vri
        s = torch.sqrt(delta.clamp_min(self.eps))
        t = torch.sqrt((vrr + vii + 2 * s).clamp_min(self.eps))
        inv_st = 1.0 / (s * t)
        wrr = (vii + s) * inv_st
        wii = (vrr + s) * inv_st
        wri = -vri * inv_st

        wrr, wii, wri = (w.view(1, -1, 1) for w in (wrr, wii, wri))
        xr_hat = wrr * cr + wri * ci
        xi_hat = wri * cr + wii * ci

        gamma_rr, gamma_ii, gamma_ri = (g.view(1, -1, 1) for g in (self.gamma_rr, self.gamma_ii, self.gamma_ri))
        beta_r, beta_i = self.beta_r.view(1, -1, 1), self.beta_i.view(1, -1, 1)
        out_r = gamma_rr * xr_hat + gamma_ri * xi_hat + beta_r
        out_i = gamma_ri * xr_hat + gamma_ii * xi_hat + beta_i

        return torch.complex(out_r, out_i)
