from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..activations import get_activation
from .im2col import col2im, im2col


@dataclass
class Conv2DConfig:
    c_in: int
    c_out: int
    kernel_size: Tuple[int, int]
    stride: int = 1
    padding: str = "valid"
    activation: str = "relu"
    init: str = "he"


class Conv2DLayer:
    layer_type = "conv2d"
    has_params = True

    def __init__(self, cfg: Conv2DConfig) -> None:
        self.cfg = cfg
        k_h, k_w = cfg.kernel_size
        fan_in = cfg.c_in * k_h * k_w
        fan_out = cfg.c_out * k_h * k_w
        std = np.sqrt(2.0 / fan_in) if cfg.init == "he" else np.sqrt(2.0 / (fan_in + fan_out))
        self.K = (np.random.randn(cfg.c_out, cfg.c_in, k_h, k_w) * std).astype(np.float32)
        self.b = np.zeros(cfg.c_out, dtype=np.float32)
        self.dK: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self.X_padded: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self._pad_hw: tuple[int, int] = (0, 0)

    def _compute_padding(self, x: np.ndarray) -> tuple[int, int]:
        if self.cfg.padding != "same":
            return 0, 0
        _, h, w = x.shape
        k_h, k_w = self.cfg.kernel_size
        pad_h = max((h - 1) * self.cfg.stride + k_h - h, 0) // 2
        pad_w = max((w - 1) * self.cfg.stride + k_w - w, 0) // 2
        return pad_h, pad_w

    def _pad(self, x: np.ndarray) -> np.ndarray:
        pad_h, pad_w = self._compute_padding(x)
        self._pad_hw = (pad_h, pad_w)
        if pad_h == 0 and pad_w == 0:
            return x
        return np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x_p = self._pad(x)
        self.X_padded = x_p
        c_out, _, k_h, k_w = self.K.shape
        _, h, w = x_p.shape
        out_h = (h - k_h) // self.cfg.stride + 1
        out_w = (w - k_w) // self.cfg.stride + 1

        x_col = im2col(x_p, k_h, k_w, self.cfg.stride)
        W_row = self.K.reshape(c_out, -1)
        z = W_row @ x_col + self.b.reshape(-1, 1)

        self.Z = z.reshape(c_out, out_h, out_w)
        act = get_activation(self.cfg.activation)
        return act.forward(self.Z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X_padded is None or self.Z is None:
            raise RuntimeError("Conv2DLayer.backward called before forward.")
        act = get_activation(self.cfg.activation)
        dZ = d_out * act.derivative(self.Z)

        c_out, c_in, k_h, k_w = self.K.shape
        _, h_p, w_p = self.X_padded.shape

        x_col = im2col(self.X_padded, k_h, k_w, self.cfg.stride)
        dZ_flat = dZ.reshape(c_out, -1)

        self.dK = (dZ_flat @ x_col.T).reshape(c_out, c_in, k_h, k_w)
        self.db = dZ_flat.sum(axis=1)

        W_row = self.K.reshape(c_out, -1)
        dX_col = W_row.T @ dZ_flat
        dX_p = col2im(dX_col, self.X_padded.shape, k_h, k_w, self.cfg.stride)

        pad_h, pad_w = self._pad_hw
        if pad_h == 0 and pad_w == 0:
            return dX_p
        return dX_p[:, pad_h : h_p - pad_h, pad_w : w_p - pad_w]

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.K, self.b

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dK, self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        self.K = w.astype(np.float32)
        self.b = b.astype(np.float32)

    def output_shape(self, input_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        c_in, h, w = input_shape
        k_h, k_w = self.cfg.kernel_size
        pad_h, pad_w = self._compute_padding(np.zeros((c_in, h, w), dtype=np.float32))
        out_h = (h + 2 * pad_h - k_h) // self.cfg.stride + 1
        out_w = (w + 2 * pad_w - k_w) // self.cfg.stride + 1
        return (self.cfg.c_out, out_h, out_w)
