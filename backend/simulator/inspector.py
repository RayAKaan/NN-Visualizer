from __future__ import annotations

from typing import Dict, List

import numpy as np

from .graph_engine import NetworkGraph


def _histogram(values: np.ndarray, bins: int = 12) -> Dict:
    counts, edges = np.histogram(values, bins=bins)
    return {"bins": [float(e) for e in edges.tolist()], "counts": [int(c) for c in counts.tolist()]}


def weight_inspection(graph: NetworkGraph, layer_index: int) -> Dict:
    layer_slot = layer_index + 1
    param_idx = graph.param_index_by_layer.get(layer_slot, layer_index)
    if param_idx < 0 or param_idx >= len(graph.weights):
        raise IndexError("layer_index out of range")
    w = graph.weights[param_idx]
    b = graph.biases[param_idx]
    return {
        "layer_index": layer_index,
        "weight_matrix": w.tolist(),
        "bias_vector": b.tolist(),
        "shape": [int(w.shape[0]), int(w.shape[1])] if w.ndim == 2 else [int(s) for s in w.shape],
        "stats": {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "l2_norm": float(np.linalg.norm(w)),
            "sparsity": float(np.mean(w == 0.0)),
        },
        "histogram": _histogram(w.flatten()),
    }


def activation_inspection(graph: NetworkGraph, layer_index: int, input_vec: List[float]) -> Dict:
    graph.forward(np.asarray(input_vec, dtype=np.float32))
    layer_slot = layer_index + 1
    if layer_slot < 0 or layer_slot >= len(graph.layer_instances):
        raise IndexError("layer_index out of range")
    layer = graph.layer_instances[layer_slot]
    pre = layer.Z if hasattr(layer, "Z") else None
    post = graph.activations[layer_slot]
    act_name = (graph.layers[layer_slot].activation or "linear").lower() if layer_slot < len(graph.layers) else "linear"
    post_flat = post.reshape(-1)
    dead = [int(i) for i, v in enumerate(post_flat) if act_name == "relu" and v == 0.0]
    return {
        "layer_index": layer_index,
        "pre_activation": pre.tolist() if pre is not None else None,
        "post_activation": post.tolist() if hasattr(post, "tolist") else post,
        "dead_neurons": dead,
        "activation_name": act_name,
        "histogram": _histogram(post_flat),
    }
