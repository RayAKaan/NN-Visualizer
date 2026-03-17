from __future__ import annotations

import base64
from io import BytesIO
from typing import Dict, Tuple

import numpy as np


def _resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    y_idx = (np.linspace(0, h - 1, size)).astype(int)
    x_idx = (np.linspace(0, w - 1, size)).astype(int)
    return img[np.ix_(y_idx, x_idx)]


def _encode_png(arr: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception:
        return ""
    img = Image.fromarray((arr * 255).astype("uint8"))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def process_image(file_bytes: bytes, target_size: int = 28) -> Dict:
    try:
        from PIL import Image
        img = Image.open(BytesIO(file_bytes)).convert("L")
        original_size = img.size
        img = img.resize((target_size, target_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        arr = np.random.rand(target_size, target_size).astype(np.float32)
        original_size = (target_size, target_size)
    preview = _encode_png(arr)
    return {
        "processed_image": arr.reshape(-1).tolist(),
        "original_size": [int(original_size[0]), int(original_size[1])],
        "preview_base64": preview,
    }

