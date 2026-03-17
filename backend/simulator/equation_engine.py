from __future__ import annotations

from typing import Dict, List

import numpy as np

from .activations import activation_plot_points, get_activation
from .graph_engine import NetworkGraph


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def layer_equations(graph: NetworkGraph, layer_index: int, include_numeric: bool = True) -> Dict:
    if layer_index < 0 or layer_index >= len(graph.layers) - 1:
        raise IndexError("layer_index out of range")

    layer_slot = layer_index + 1
    layer = graph.layers[layer_slot]
    act_name = (layer.activation or "linear").lower()
    activation = get_activation(act_name)

    param_idx = graph.param_index_by_layer.get(layer_slot)
    if param_idx is None or param_idx >= len(graph.weights):
        return {
            "layer_index": layer_index,
            "layer_type": layer.layer_type,
            "activation": act_name,
            "generic_equations": {},
            "activation_plot": {
                "name": activation.name,
                "formula": activation.formula,
                "domain": [-3, 3],
                "points": activation_plot_points(act_name),
            },
            "dimensions": {},
            "weight_stats": {},
            "numeric_equations": [],
            "message": "No parameterized equation available for this layer type.",
        }

    w = graph.weights[param_idx]
    b = graph.biases[param_idx]

    generic = {
        "pre_activation": f"z^[{layer_index + 1}] = W^[{layer_index + 1}] * a^[{layer_index}] + b^[{layer_index + 1}]",
        "activation": f"a^[{layer_index + 1}] = {activation.formula}",
        "per_neuron": "z_j = S_i w_ji * a_i + b_j",
    }

    numeric_equations: List[Dict] = []
    if include_numeric and graph.activations:
        a_prev = graph.activations[layer_slot - 1].reshape(-1)
        z = w.reshape(w.shape[0], -1) @ a_prev + b
        a = activation.forward(z)
        for j in range(len(z)):
            terms = [f"({_fmt(w.reshape(w.shape[0], -1)[j, i])})({_fmt(a_prev[i])})" for i in range(len(a_prev))]
            pre_eq = f"z{j + 1} = " + " + ".join(terms) + f" + ({_fmt(b[j])}) = {_fmt(z[j])}"
            act_eq = f"a{j + 1} = {act_name}({_fmt(z[j])}) = {_fmt(a[j])}"
            numeric_equations.append(
                {
                    "neuron_index": j,
                    "pre_activation_eq": pre_eq,
                    "activation_eq": act_eq,
                    "pre_activation_value": float(z[j]),
                    "activation_value": float(a[j]),
                    "is_dead": bool(act_name == "relu" and a[j] == 0.0),
                }
            )

    return {
        "layer_index": layer_index,
        "layer_type": layer.layer_type,
        "activation": act_name,
        "generic_equations": generic,
        "activation_plot": {
            "name": activation.name,
            "formula": activation.formula,
            "domain": [-3, 3],
            "points": activation_plot_points(act_name),
        },
        "dimensions": {
            "W_shape": [int(s) for s in w.shape],
            "b_shape": [int(b.shape[0])],
            "input_dim": int(w.reshape(w.shape[0], -1).shape[1]),
            "output_dim": int(w.shape[0]),
            "param_count": int(w.size + b.size),
        },
        "weight_stats": {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
        },
        "numeric_equations": numeric_equations,
    }
