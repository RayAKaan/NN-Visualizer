from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from ..backward_engine import backward_full
from ..visualization.rendering import render_heatmap_base64


def deep_shap(
    graph: NetworkGraph,
    input_vec: List[float],
    target_class: int,
    baseline_samples: int = 50,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    # baseline is mean of random noise samples
    baselines = np.random.rand(baseline_samples, x.size).astype(np.float32) * 0.0
    baseline = baselines.mean(axis=0)

    target = np.zeros_like(graph.forward(x))
    if target.size > target_class:
        target[target_class] = 1.0

    back = backward_full(graph, x.tolist(), target.tolist(), "mse", 0.0)
    grad = np.asarray(back.get("input_gradient", np.zeros_like(x)), dtype=np.float32)
    shap_values = (x - baseline) * grad
    heatmap = render_heatmap_base64(shap_values.reshape(1, -1))

    return {
        "shap_values": shap_values.tolist(),
        "shap_base64": heatmap,
        "base_value": float(graph.forward(baseline)[target_class]) if graph.forward(baseline).size > target_class else 0.0,
        "output_value": float(graph.forward(x)[target_class]) if graph.forward(x).size > target_class else 0.0,
        "force_plot_data": {
            "positive_features": [],
            "negative_features": [],
        },
        "summary_importance": [],
    }
