from __future__ import annotations

from typing import Dict, List
import numpy as np


def apply_transform(img: np.ndarray, transform: Dict) -> np.ndarray:
    t = transform.get("type")
    if t == "noise":
        sigma = float(transform.get("sigma", 0.1))
        return img + np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
    if t == "flip_h":
        return np.flip(img, axis=1)
    if t == "flip_v":
        return np.flip(img, axis=0)
    if t == "rotate90":
        k = int(transform.get("k", 1)) % 4
        return np.rot90(img, k)
    if t == "crop":
        ratio = float(transform.get("ratio", 0.8))
        h, w = img.shape
        ch, cw = int(h * ratio), int(w * ratio)
        h0 = (h - ch) // 2
        w0 = (w - cw) // 2
        cropped = img[h0:h0+ch, w0:w0+cw]
        # resize back (nearest)
        y_idx = (np.linspace(0, ch - 1, h)).astype(int)
        x_idx = (np.linspace(0, cw - 1, w)).astype(int)
        return cropped[np.ix_(y_idx, x_idx)]
    return img


def apply_pipeline(img: np.ndarray, pipeline: List[Dict]) -> np.ndarray:
    out = img
    for t in pipeline:
        out = apply_transform(out, t)
    return out
