from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from ..backward_engine import backward_full
from ..visualization.rendering import render_heatmap_base64


def integrated_gradients(
    graph: NetworkGraph,
    input_vec: List[float],
    target_class: int,
    n_steps: int = 50,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    baseline = np.zeros_like(x)
    scaled_inputs = [baseline + (float(k) / n_steps) * (x - baseline) for k in range(1, n_steps + 1)]

    grads = []
    for xi in scaled_inputs:
        target = np.zeros_like(graph.forward(xi))
        if target.size > target_class:
            target[target_class] = 1.0
        result = backward_full(graph, xi.tolist(), target.tolist(), "mse", 0.0)
        d_input = np.asarray(result.get("input_gradient", np.zeros_like(x)), dtype=np.float32)
        grads.append(d_input)

    avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
    attributions = (x - baseline) * avg_grad
    heatmap = render_heatmap_base64(attributions.reshape(1, -1))

    return {
        "attributions": attributions.tolist(),
        "attributions_base64": heatmap,
        "convergence_delta": float(np.abs(attributions.sum() - (graph.forward(x)[target_class] - graph.forward(baseline)[target_class]))),
        "baseline_output": float(graph.forward(baseline)[target_class]) if graph.forward(baseline).size > target_class else 0.0,
        "input_output": float(graph.forward(x)[target_class]) if graph.forward(x).size > target_class else 0.0,
        "attribution_sum": float(attributions.sum()),
        "top_features": [],
    }
