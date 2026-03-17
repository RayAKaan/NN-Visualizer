from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PoolConfig:
    pool_size: int = 2
    stride: int | None = None


class MaxPool2DLayer:
    layer_type = "maxpool2d"
    has_params = False

    def __init__(self, cfg: PoolConfig) -> None:
        self.cfg = cfg
        self.max_indices: np.ndarray | None = None
        self.input_shape: tuple[int, int, int] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        c, h, w = x.shape
        k = self.cfg.pool_size
        s = self.cfg.stride or k
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
        self.max_indices = np.zeros_like(out, dtype=np.int32)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i * s : i * s + k, j * s : j * s + k]
                flat = patch.reshape(c, -1)
                idx = np.argmax(flat, axis=1)
                out[:, i, j] = flat[np.arange(c), idx]
                self.max_indices[:, i, j] = idx
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.input_shape is None or self.max_indices is None:
            raise RuntimeError("MaxPool2DLayer.backward called before forward.")
        c, h, w = self.input_shape
        k = self.cfg.pool_size
        s = self.cfg.stride or k
        out_h, out_w = d_out.shape[1], d_out.shape[2]
        dX = np.zeros((c, h, w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                idx = self.max_indices[:, i, j]
                for ch in range(c):
                    r = idx[ch] // k
                    c_idx = idx[ch] % k
                    dX[ch, i * s + r, j * s + c_idx] += d_out[ch, i, j]
        return dX

    def params(self) -> tuple[None, None]:
        return None, None

    def grads(self) -> tuple[None, None]:
        return None, None

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        return


class AvgPool2DLayer:
    layer_type = "avgpool2d"
    has_params = False

    def __init__(self, cfg: PoolConfig) -> None:
        self.cfg = cfg
        self.input_shape: tuple[int, int, int] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        c, h, w = x.shape
        k = self.cfg.pool_size
        s = self.cfg.stride or k
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        out = np.zeros((c, out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i * s : i * s + k, j * s : j * s + k]
                out[:, i, j] = patch.mean(axis=(1, 2))
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.input_shape is None:
            raise RuntimeError("AvgPool2DLayer.backward called before forward.")
        c, h, w = self.input_shape
        k = self.cfg.pool_size
        s = self.cfg.stride or k
        out_h, out_w = d_out.shape[1], d_out.shape[2]
        dX = np.zeros((c, h, w), dtype=np.float32)
        scale = 1.0 / (k * k)
        for i in range(out_h):
            for j in range(out_w):
                dX[:, i * s : i * s + k, j * s : j * s + k] += d_out[:, i, j][:, None, None] * scale
        return dX

    def params(self) -> tuple[None, None]:
        return None, None

    def grads(self) -> tuple[None, None]:
        return None, None

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        return
