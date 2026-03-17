from __future__ import annotations

from typing import Dict, List

import numpy as np

from .graph_engine import NetworkGraph
from .visualization.rendering import render_gray


def _find_sequence_layer(graph: NetworkGraph) -> int:
    for idx in range(1, len(graph.layer_instances)):
        if graph.layers[idx].layer_type in {"rnn", "lstm"}:
            return idx
    return -1


def sequence_step(graph: NetworkGraph, sequence: List[List[float]], timestep: int) -> Dict:
    seq = np.asarray(sequence, dtype=np.float32)
    layer_idx = _find_sequence_layer(graph)
    if layer_idx < 0:
        raise ValueError("No sequence layer found")
    layer = graph.layer_instances[layer_idx]
    output = layer.forward(seq)
    t = int(timestep)
    if t < 0 or t >= seq.shape[0]:
        raise IndexError("timestep out of range")

    result = {
        "timestep": t,
        "input_t": seq[t].tolist(),
        "new_hidden": (layer.H[t] if hasattr(layer, "H") else output).tolist(),
        "hidden_history": [{"t": i + 1, "h": layer.H[i].tolist()} for i in range(t + 1)]
        if hasattr(layer, "H")
        else [],
    }

    if graph.layers[layer_idx].layer_type == "lstm":
        result["previous_hidden"] = layer.H[t - 1].tolist() if t > 0 else [0.0] * layer.hidden_dim
        result["previous_cell"] = layer.C[t - 1].tolist() if t > 0 else [0.0] * layer.hidden_dim
        result["new_cell"] = layer.C[t].tolist()
        if layer.gates is not None:
            result["gates"] = {
                "forget": {"values": layer.gates["f"][t].tolist(), "equation": "f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)"},
                "input": {"values": layer.gates["i"][t].tolist(), "equation": "i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)"},
                "output": {"values": layer.gates["o"][t].tolist(), "equation": "o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)"},
                "candidate": {"values": layer.gates["g"][t].tolist(), "equation": "c~_t = tanh(W_c · [h_{t-1}, x_t] + b_c)"},
            }
    else:
        result["previous_hidden"] = layer.H[t - 1].tolist() if t > 0 else [0.0] * layer.hidden_dim

    return result


def sequence_full(graph: NetworkGraph, sequence: List[List[float]]) -> Dict:
    seq = np.asarray(sequence, dtype=np.float32)
    layer_idx = _find_sequence_layer(graph)
    if layer_idx < 0:
        raise ValueError("No sequence layer found")
    layer = graph.layer_instances[layer_idx]
    output = layer.forward(seq)
    data = {
        "all_hidden_states": layer.H.tolist() if hasattr(layer, "H") else [],
        "all_cell_states": layer.C.tolist() if hasattr(layer, "C") else [],
        "all_gate_values": layer.gates if hasattr(layer, "gates") else {},
        "final_output": output.tolist() if hasattr(output, "tolist") else output,
    }
    # attention weights if present
    for idx in range(1, len(graph.layer_instances)):
        if graph.layers[idx].layer_type == "attention":
            attn = graph.layer_instances[idx]
            if attn.attn_weights is not None:
                data["all_attention_weights"] = attn.attn_weights.tolist()
                data["attention_heatmap_base64"] = render_gray(attn.attn_weights)
            break
    return data
