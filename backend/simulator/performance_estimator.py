from __future__ import annotations

from typing import Dict

from .graph_engine import NetworkGraph


def estimate_performance(graph: NetworkGraph, n_samples: int, batch_size: int, optimizer: str) -> Dict:
    fwd = graph.flops_per_sample
    bwd = int(fwd * 2)
    flops_epoch = (fwd + bwd) * n_samples

    params = sum(w.size + b.size for w, b in zip(graph.weights, graph.biases))
    param_bytes = params * 4
    activations_bytes = sum(layer.neurons for layer in graph.layers) * batch_size * 4
    gradients_bytes = params * 4
    opt_factor = 0
    if optimizer in {"sgd_momentum", "momentum"}:
        opt_factor = 1
    if optimizer == "adam":
        opt_factor = 2
    opt_bytes = params * 4 * opt_factor
    total_bytes = param_bytes + activations_bytes + gradients_bytes + opt_bytes

    return {
        "flops_per_sample_fwd": fwd,
        "flops_per_sample_bwd": bwd,
        "flops_per_epoch": flops_epoch,
        "memory_bytes": {
            "parameters": param_bytes,
            "activations": activations_bytes,
            "gradients": gradients_bytes,
            "optimizer_state": opt_bytes,
            "total": total_bytes,
        },
        "estimated_time_ms": {
            "epoch": flops_epoch / 1_000_000.0,
        },
    }

