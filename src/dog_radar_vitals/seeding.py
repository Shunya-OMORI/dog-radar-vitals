"""乱数シードの一元管理。学習を担う全trainerがここを経由する。"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Python/NumPy/PyTorch(CPU・CUDA)の乱数シードを固定する。

    `deterministic=True` でcuDNNを決定的アルゴリズムに固定するが、GPU・ドライバ・
    CUDAバージョンが変わるとビット単位の再現は保証されない点に注意（学習曲線の
    再現性は概ね得られるが、査読者の環境で完全一致は期待しない）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """DataLoaderのshuffle順を固定するための専用Generator。"""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
