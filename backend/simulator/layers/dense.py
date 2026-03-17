from __future__ import annotations

import numpy as np

from ..activations import get_activation


class DenseLayer:
    layer_type = "dense"
    has_params = True

    def __init__(self, in_dim: int, out_dim: int, activation: str | None = None, init: str | None = None) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_name = (activation or "linear").lower()
        self.init = init
        self.W = self._init_weights((out_dim, in_dim), init)
        self.b = self._init_bias((out_dim,), init)
        self.X: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    @staticmethod
    def _init_weights(shape: tuple[int, int], init: str | None) -> np.ndarray:
        init_name = (init or "xavier").lower()
        fan_out, fan_in = shape
        if init_name in {"xavier", "glorot"}:
            std = np.sqrt(2.0 / (fan_in + fan_out))
        elif init_name == "he":
            std = np.sqrt(2.0 / fan_in)
        elif init_name == "lecun":
            std = np.sqrt(1.0 / fan_in)
        elif init_name == "zeros":
            return np.zeros(shape, dtype=np.float32)
        else:
            std = 0.01
        return np.random.normal(0.0, std, size=shape).astype(np.float32)

    @staticmethod
    def _init_bias(shape: tuple[int], init: str | None) -> np.ndarray:
        init_name = (init or "zeros").lower()
        if init_name == "random":
            return np.random.normal(0.0, 0.01, size=shape).astype(np.float32)
        return np.zeros(shape, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1).astype(np.float32)
        self.X = x
        z = self.W @ x + self.b
        self.Z = z
        act = get_activation(self.activation_name)
        return act.forward(z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X is None or self.Z is None:
            raise RuntimeError("DenseLayer.backward called before forward.")
        act = get_activation(self.activation_name)
        dZ = d_out * act.derivative(self.Z)
        self.dW = np.outer(dZ, self.X)
        self.db = dZ.copy()
        dX = self.W.T @ dZ
        return dX

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.W, self.b

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dW, self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        self.W = w.astype(np.float32)
        self.b = b.astype(np.float32)

