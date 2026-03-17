from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple

import numpy as np


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    norm = (img - mn) / (mx - mn)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def render_gray(img: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    arr = _to_uint8(img)
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        im = Image.fromarray(arr)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_heatmap_overlay(base: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> Tuple[str, str]:
    try:
        from PIL import Image
    except Exception:
        return "", ""
    base_img = _to_uint8(base)
    heat_img = _to_uint8(heat)
    if base_img.ndim == 2:
        base_rgb = np.stack([base_img] * 3, axis=-1)
    else:
        base_rgb = base_img
    heat_rgb = np.zeros_like(base_rgb)
    heat_rgb[..., 0] = heat_img
    overlay = (1 - alpha) * base_rgb + alpha * heat_rgb
    overlay = overlay.clip(0, 255).astype(np.uint8)
    base_png = Image.fromarray(base_rgb.astype(np.uint8))
    heat_png = Image.fromarray(overlay.astype(np.uint8))
    buf1 = BytesIO()
    buf2 = BytesIO()
    base_png.save(buf1, format="PNG")
    heat_png.save(buf2, format="PNG")
    return (
        "data:image/png;base64," + base64.b64encode(buf1.getvalue()).decode("ascii"),
        "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode("ascii"),
    )


def resize_nearest(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    new_h, new_w = size
    y_idx = (np.linspace(0, h - 1, new_h)).astype(int)
    x_idx = (np.linspace(0, w - 1, new_w)).astype(int)
    return img[np.ix_(y_idx, x_idx)]

def render_heatmap_base64(img: np.ndarray) -> str:
    return render_gray(img)

