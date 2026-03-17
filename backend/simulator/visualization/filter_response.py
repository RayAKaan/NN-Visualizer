from __future__ import annotations

from typing import Dict

import numpy as np

from ..dataset_manager import dataset_manager
from ..graph_engine import NetworkGraph
from .rendering import render_heatmap_overlay


def compute_filter_response(
    graph: NetworkGraph,
    dataset_id: str,
    layer_index: int,
    filter_index: int,
    n_samples: int,
) -> Dict:
    data = dataset_manager.get(dataset_id)
    samples = data.get("train", {}).get("x") or []
    if not samples:
        return {"filter_index": filter_index, "top_activating_samples": []}
    n = min(int(n_samples), len(samples))
    scores = []
    for i in range(n):
        x = np.asarray(samples[i], dtype=np.float32)
        graph.forward(x)
        layer_slot = layer_index + 1
        fmap = graph.activations[layer_slot][filter_index]
        scores.append((float(np.max(fmap)), i, fmap))
    scores.sort(key=lambda s: s[0], reverse=True)
    top = []
    for score, idx, fmap in scores[: min(10, len(scores))]:
        base = np.asarray(samples[idx], dtype=np.float32)
        if base.ndim == 3:
            base = base[0]
        heat_base64, overlay = render_heatmap_overlay(base, fmap)
        top.append(
            {
                "sample_index": idx,
                "image_base64": heat_base64,
                "max_activation": score,
                "activation_map_base64": overlay,
            }
        )
    return {
        "filter_index": filter_index,
        "kernel": graph.layer_instances[layer_index + 1].K[filter_index, 0].tolist() if hasattr(graph.layer_instances[layer_index + 1], "K") else [],
        "kernel_description": "kernel",
        "top_activating_samples": top,
    }
