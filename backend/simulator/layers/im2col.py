from __future__ import annotations

import numpy as np


def im2col(
    x: np.ndarray, k_h: int, k_w: int, stride: int
) -> np.ndarray:
    C, H, W = x.shape
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1

    shape = (C, out_h, out_w, k_h, k_w)
    strides = (
        x.strides[0],
        x.strides[1] * stride,
        x.strides[2] * stride,
        x.strides[1],
        x.strides[2],
    )

    windows = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )
    return windows.transpose(0, 3, 4, 1, 2).reshape(C * k_h * k_w, out_h * out_w)


def col2im(
    cols: np.ndarray, x_shape: tuple[int, int, int], k_h: int, k_w: int, stride: int
) -> np.ndarray:
    C, H, W = x_shape
    out_h = (H - k_h) // stride + 1
    out_w = (W - k_w) // stride + 1

    windows = cols.reshape(C, k_h, k_w, out_h, out_w)
    x = np.zeros(x_shape, dtype=cols.dtype)

    out_h_idx = np.arange(out_h)
    out_w_idx = np.arange(out_w)

    for di in range(k_h):
        for dj in range(k_w):
            h_start = out_h_idx * stride + di
            w_start = out_w_idx * stride + dj
            x[:, h_start[:, np.newaxis], w_start[np.newaxis, :]] += windows[:, di, dj, :, :]

    return x
