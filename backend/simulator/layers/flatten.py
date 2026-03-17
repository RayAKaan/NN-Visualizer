from __future__ import annotations

import numpy as np


class FlattenLayer:
    layer_type = "flatten"
    has_params = False

    def __init__(self) -> None:
        self.input_shape: tuple[int, ...] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(-1).astype(np.float32)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise RuntimeError("FlattenLayer.backward called before forward.")
        return d_out.reshape(self.input_shape).astype(np.float32)

    def params(self) -> tuple[None, None]:
        return None, None

    def grads(self) -> tuple[None, None]:
        return None, None

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        return
