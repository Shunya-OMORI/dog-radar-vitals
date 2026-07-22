"""学習率スケジューラの構築。epoch単位で呼び出すシンプルな実装に絞る。

configで `train.scheduler` を指定しない限りスケジューラなし（定数LR）のままで、
既存の全configは無変更で動く。
"""
from __future__ import annotations

import math

import torch


class WarmupCosineScheduler:
    """線形ウォームアップ後にコサイン減衰する、epoch単位のスケジューラ。"""

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, warmup_epochs: int, total_epochs: int) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self._epoch = 0

    def step(self) -> None:
        if self._epoch < self.warmup_epochs:
            lr = self.peak_lr * (self._epoch + 1) / self.warmup_epochs
        else:
            progress = (self._epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.peak_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._epoch += 1

    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict, total_epochs: int):
    sched_cfg = train_cfg.get("scheduler", {"type": "none"})
    sched_type = sched_cfg.get("type", "none")
    if sched_type == "none":
        return None
    if sched_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            peak_lr=train_cfg["lr"],
            warmup_epochs=sched_cfg["warmup_epochs"],
            total_epochs=total_epochs,
        )
    raise ValueError(f"unknown scheduler type '{sched_type}'")
