from __future__ import annotations

from typing import List, Dict
import numpy as np

from ..visualization.rendering import render_gray
from .transforms import apply_pipeline


def preview_augmentations(input_vec: List[float], input_shape: List[int] | None, n_samples: int, pipeline: List[Dict] | None = None) -> List[str]:
    x = np.asarray(input_vec, dtype=np.float32)
    if input_shape and len(input_shape) >= 2:
        h, w = input_shape[-2], input_shape[-1]
        if h * w == x.size:
            base = x.reshape(h, w)
        else:
            base = x.reshape(1, -1)
    else:
        base = x.reshape(1, -1)

    pipeline = pipeline or []
    samples = []
    for _ in range(n_samples):
        img = base.copy()
        if img.ndim == 2:
            img = apply_pipeline(img, pipeline)
        samples.append(render_gray(img))
    return samples
