from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .activations import get_activation
from .graph_engine import NetworkGraph


def _fmt_num(v: float) -> str:
    return f"{v:.3f}"


def run_forward_full(graph: NetworkGraph, input_vec: List[float]) -> Tuple[List[dict], List[float], Dict[int, List[float]]]:
    x = np.asarray(input_vec, dtype=np.float32)
    steps: List[dict] = []
    activations = [x]
    layer_outputs: Dict[int, List[float]] = {}
    step_index = 0

    dense_only = all(layer.layer_type in {"input", "dense", "output"} for layer in graph.layers)
    if dense_only:
        for layer_idx in range(1, len(graph.layers)):
            w = graph.weights[layer_idx - 1]
            b = graph.biases[layer_idx - 1]
            z_raw = w @ activations[-1]

            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx - 1,
                    "operation": "matmul",
                    "description": f"Layer {layer_idx}: Matrix Multiplication",
                    "equation_text": "z = W·a",
                    "input_values": activations[-1].tolist(),
                    "output_values": z_raw.tolist(),
                    "weights_snapshot": w.tolist(),
                }
            )
            step_index += 1

            z = z_raw + b
            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx - 1,
                    "operation": "bias_add",
                    "description": f"Layer {layer_idx}: Bias Addition",
                    "equation_text": "z = z + b",
                    "input_values": z_raw.tolist(),
                    "output_values": z.tolist(),
                    "bias_snapshot": b.tolist(),
                }
            )
            step_index += 1

            act_name = (graph.layers[layer_idx].activation or "linear").lower()
            a = get_activation(act_name).forward(z)
            steps.append(
                {
                    "step_index": step_index,
                    "layer_index": layer_idx - 1,
                    "operation": "activation",
                    "description": f"Layer {layer_idx}: {act_name} Activation",
                    "activation_name": act_name,
                    "equation_text": f"a = {act_name}(z)",
                    "input_values": z.tolist(),
                    "output_values": a.tolist(),
                }
            )
            step_index += 1
            activations.append(a)
            layer_outputs[layer_idx - 1] = a.tolist()

        graph.pre_activations = [w @ activations[i] + graph.biases[i] for i, w in enumerate(graph.weights)]
        graph.activations = activations
        return steps, activations[-1].tolist(), layer_outputs

    current = x
    graph.pre_activations = []
    graph.activations = [x]
    for layer_idx in range(1, len(graph.layers)):
        layer = graph.layer_instances[layer_idx]
        current = layer.forward(current)
        graph.activations.append(current)
        if hasattr(layer, "Z") and getattr(layer, "Z") is not None:
            graph.pre_activations.append(layer.Z)
        steps.append(
            {
                "step_index": step_index,
                "layer_index": layer_idx - 1,
                "operation": "forward",
                "description": f"Layer {layer_idx}: {graph.layers[layer_idx].layer_type} forward",
                "input_values": activations[-1].reshape(-1).tolist(),
                "output_values": current.reshape(-1).tolist(),
            }
        )
        step_index += 1
        activations.append(current)
        layer_outputs[layer_idx - 1] = current.reshape(-1).tolist()

    return steps, activations[-1].reshape(-1).tolist(), layer_outputs


def run_forward_step(graph: NetworkGraph, input_vec: List[float], step_index: int) -> dict:
    steps, final_output, layer_outputs = run_forward_full(graph, input_vec)
    if step_index < 0 or step_index >= len(steps):
        raise IndexError("step_index out of range")

    completed_layers = sorted(
        {s["layer_index"] for s in steps[: step_index + 1] if s["operation"] in {"activation", "forward"}}
    )
    active_layer = steps[step_index]["layer_index"]
    partial = {str(k): v for k, v in layer_outputs.items() if k in completed_layers}

    return {
        "step": steps[step_index],
        "completed_layers": completed_layers,
        "active_layer": active_layer,
        "partial_activations": partial,
    }
