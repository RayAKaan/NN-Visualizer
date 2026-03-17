from __future__ import annotations

import numpy as np


class ResidualLayer:
    layer_type = "residual"
    has_params = False

    def __init__(self) -> None:
        self.last_input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_input = x
        return x

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out

    def params(self) -> tuple[None, None]:
        return None, None

    def grads(self) -> tuple[None, None]:
        return None, None

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        return

