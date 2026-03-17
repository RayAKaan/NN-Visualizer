from __future__ import annotations

from typing import List
import numpy as np

from ..visualization.rendering import render_gray


def random_samples(n_samples: int, size: int = 28) -> List[str]:
    images = []
    for _ in range(n_samples):
        img = np.random.rand(size, size).astype(np.float32)
        images.append(render_gray(img))
    return images
