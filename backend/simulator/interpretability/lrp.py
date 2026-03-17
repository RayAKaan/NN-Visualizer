from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from ..visualization.rendering import render_heatmap_base64


def lrp_explain(
    graph: NetworkGraph,
    input_vec: List[float],
    target_class: int,
    epsilon: float = 1e-2,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    graph.forward(x)

    # dense-only path
    if not graph.weights:
        return {"input_relevance": [], "input_relevance_base64": "", "relevance_per_layer": [], "conservation_error": 0.0}

    activations = graph.activations
    weights = graph.weights

    y = graph.activations[-1].reshape(-1)
    R = np.zeros_like(y)
    if y.size > target_class:
        R[target_class] = y[target_class]

    relevance_per_layer = []
    for layer_idx in reversed(range(len(weights))):
        a_prev = activations[layer_idx]
        w = weights[layer_idx]
        z = w @ a_prev
        z = z + epsilon * np.sign(z)
        s = R / (z + 1e-12)
        c = w.T @ s
        R = a_prev * c
        relevance_per_layer.append({"layer": layer_idx, "relevance": R.tolist(), "total_relevance": float(np.sum(R))})

    relevance_per_layer = list(reversed(relevance_per_layer))
    heatmap = render_heatmap_base64(R.reshape(1, -1))
    return {
        "relevance_per_layer": relevance_per_layer,
        "input_relevance": R.tolist(),
        "input_relevance_base64": heatmap,
        "conservation_error": float(np.abs(np.sum(R) - y[target_class])) if y.size > target_class else 0.0,
    }
