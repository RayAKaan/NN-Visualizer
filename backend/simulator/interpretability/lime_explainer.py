from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from ..visualization.rendering import render_gray


def _segment_grid(h: int, w: int, n_segments: int) -> np.ndarray:
    # simple grid segmentation
    grid_size = int(np.sqrt(n_segments))
    grid_size = max(1, grid_size)
    seg = np.zeros((h, w), dtype=np.int32)
    hs = h // grid_size
    ws = w // grid_size
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            h0 = i * hs
            w0 = j * ws
            h1 = h if i == grid_size - 1 else (i + 1) * hs
            w1 = w if j == grid_size - 1 else (j + 1) * ws
            seg[h0:h1, w0:w1] = idx
            idx += 1
    return seg


def lime_explain(
    graph: NetworkGraph,
    input_vec: List[float],
    input_shape: List[int] | None,
    target_class: int,
    n_samples: int = 200,
    n_superpixels: int = 16,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    if input_shape and len(input_shape) >= 2:
        h, w = input_shape[-2], input_shape[-1]
    else:
        size = int(np.sqrt(x.size))
        h, w = (size, size) if size * size == x.size else (1, x.size)

    img = x.reshape(h, w)
    seg = _segment_grid(h, w, n_superpixels)
    n_segments = int(seg.max() + 1)

    # generate perturbations
    Z = np.random.randint(0, 2, size=(n_samples, n_segments))
    preds = []
    for i in range(n_samples):
        mask = Z[i]
        pert = img.copy()
        for s in range(n_segments):
            if mask[s] == 0:
                pert[seg == s] = 0.0
        y = graph.forward(pert.reshape(-1))
        preds.append(float(y[target_class]) if y.size > target_class else 0.0)

    # weighted linear regression
    Z = np.hstack([np.ones((n_samples, 1)), Z])
    W = np.eye(n_samples)
    y = np.asarray(preds, dtype=np.float32)
    try:
        coef = np.linalg.pinv(Z.T @ W @ Z) @ (Z.T @ W @ y)
    except np.linalg.LinAlgError:
        coef = np.zeros(Z.shape[1], dtype=np.float32)

    importances = coef[1:]
    explanation = np.zeros_like(img)
    for s in range(n_segments):
        explanation[seg == s] = importances[s]

    return {
        "superpixel_map": seg.tolist(),
        "superpixel_importances": [
            {"segment_id": int(s), "importance": float(importances[s]), "sign": "positive" if importances[s] >= 0 else "negative"}
            for s in range(n_segments)
        ],
        "explanation_image_base64": render_gray(explanation),
        "local_model_r2": 0.0,
        "local_model_weights": importances.tolist(),
    }
