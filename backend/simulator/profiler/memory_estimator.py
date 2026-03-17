from __future__ import annotations

from typing import Dict, List

from ..graph_engine import NetworkGraph
from .utils import infer_layer_shapes


def estimate_memory(graph: NetworkGraph) -> Dict:
    params = sum(w.size + b.size for w, b in zip(graph.weights, graph.biases))
    params_bytes = params * 4

    shapes = infer_layer_shapes(graph.layers)
    activation_elems = 0
    per_layer: List[Dict] = []
    for idx, shape in enumerate(shapes[1:], start=1):
        elems = 1
        for dim in shape:
            elems *= dim
        activation_elems += elems
        per_layer.append(
            {
                "layer_index": idx,
                "activation_elements": int(elems),
                "activation_bytes": int(elems * 4),
            }
        )

    activations_bytes = activation_elems * 4
    gradients_bytes = params_bytes
    optimizer_bytes = params_bytes * 2

    return {
        "params": int(params),
        "params_bytes": int(params_bytes),
        "activations_bytes": int(activations_bytes),
        "gradients_bytes": int(gradients_bytes),
        "optimizer_bytes": int(optimizer_bytes),
        "per_layer": per_layer,
    }
