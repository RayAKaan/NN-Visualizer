from __future__ import annotations

from typing import Tuple
import numpy as np


def apply_dropout(a: np.ndarray, dropout_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    if dropout_rate <= 0.0:
        return a, np.ones_like(a)
    keep_prob = 1.0 - dropout_rate
    mask = (np.random.rand(*a.shape) < keep_prob).astype(np.float32)
    return (a * mask) / keep_prob, mask

