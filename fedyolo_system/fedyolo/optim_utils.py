"""Tiny shared training utilities."""

from __future__ import annotations

import torch


def set_warmup_lr(optimizer: torch.optim.Optimizer, step: int, warmup_steps: int, lr0: float) -> None:
    """Linear LR warmup from 0 -> lr0 over the first `warmup_steps` steps.

    Random-init detectors take large, unstable gradient steps on the very
    first few batches (box/DFL predictions start essentially arbitrary);
    skipping warmup is the most common cause of the loss oscillating wildly
    instead of decreasing, especially with tiny batch sizes.
    """
    lr = lr0 * min(1.0, (step + 1) / max(1, warmup_steps))
    for g in optimizer.param_groups:
        g["lr"] = lr
