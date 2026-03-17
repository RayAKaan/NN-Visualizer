from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..graph_engine import NetworkGraph
from .rendering import render_heatmap_overlay


def compute_saliency(graph: NetworkGraph, input_vec: List[float], input_shape: List[int], target_class: int) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32).reshape(input_shape)
    graph.forward(x)
    output = graph.activations[-1].reshape(-1)
    if target_class < 0 or target_class >= len(output):
        target_class = int(np.argmax(output))
    d_out = np.zeros_like(output)
    d_out[target_class] = 1.0

    grad = d_out
    for idx in reversed(range(1, len(graph.layer_instances))):
        layer = graph.layer_instances[idx]
        grad = layer.backward(grad)
    saliency = np.abs(grad).reshape(input_shape)
    base = x.reshape(input_shape)[-1] if x.ndim == 3 else x.reshape(input_shape)
    heat = saliency[-1] if saliency.ndim == 3 else saliency
    base64_input, overlay = render_heatmap_overlay(base, heat)
    return {
        "method": "gradient",
        "saliency_map": saliency.tolist(),
        "saliency_base64": base64_input,
        "overlay_base64": overlay,
        "top_pixels": [],
    }
