"""μ-law companding変換(Chen et al. 2022, 式14)。ECGの振幅分布を均し、R波（大振幅）と
P/T波（小振幅）を同じスケールで扱えるようにしたうえで、256値へ量子化してカテゴリカル分布
として学習するために使う（本ファイルの上位: `models/deep/chen2022_reconstructor.py`,
`training/chen2022_trainer.py`）。

入力は事前に[-1, 1]へmin-max正規化されている前提（`mmecg_windowing.normalization="minmax"`）。
"""
from __future__ import annotations

import numpy as np
import torch

MU = 255
N_CLASSES = MU + 1  # 256


def mu_law_encode(x: torch.Tensor, mu: int = MU) -> torch.Tensor:
    """式(14): f(x) = sign(x) * ln(1+mu|x|) / ln(1+mu)。x, 戻り値ともに[-1, 1]。"""
    return torch.sign(x) * torch.log1p(mu * x.abs()) / np.log1p(mu)


def mu_law_decode(y: torch.Tensor, mu: int = MU) -> torch.Tensor:
    """mu_law_encodeの逆変換。"""
    return torch.sign(y) * (torch.expm1(y.abs() * np.log1p(mu)) / mu)


def quantize(y: torch.Tensor, mu: int = MU) -> torch.Tensor:
    """companding後の値[-1, 1]を0..muの整数ビンへ量子化する(int64)。"""
    return ((y.clamp(-1.0, 1.0) + 1.0) / 2.0 * mu).round().long()


def dequantize(bins: torch.Tensor, mu: int = MU) -> torch.Tensor:
    """quantizeの逆（ビン中心値、companding空間の連続値[-1,1]を返す）。"""
    return bins.float() / mu * 2.0 - 1.0
