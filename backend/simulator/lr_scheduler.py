from __future__ import annotations

import math


def get_lr(base_lr: float, epoch: int, scheduler: str | None, decay_rate: float | None, step_size: int | None) -> float:
    if not scheduler:
        return base_lr
    key = scheduler.lower()
    gamma = decay_rate if decay_rate is not None else 0.5
    if key == "step":
        step = step_size if step_size else 50
        return base_lr * (gamma ** (epoch // step))
    if key == "exponential":
        return base_lr * (gamma ** epoch)
    if key == "cosine":
        t = step_size if step_size else 100
        return 0.5 * base_lr * (1 + math.cos(math.pi * epoch / t))
    return base_lr

