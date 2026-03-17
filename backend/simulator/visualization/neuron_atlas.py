from __future__ import annotations

from typing import Dict

import numpy as np

from ..dataset_manager import dataset_manager
from ..graph_engine import NetworkGraph


def compute_neuron_atlas(graph: NetworkGraph, dataset_id: str, layer_index: int, n_samples: int) -> Dict:
    data = dataset_manager.get(dataset_id)
    samples = data.get("train", {}).get("x") or []
    if not samples:
        return {"layer_index": layer_index, "neurons": []}
    n = min(int(n_samples), len(samples))
    activations = []
    for i in range(n):
        x = np.asarray(samples[i], dtype=np.float32)
        graph.forward(x)
        layer_slot = layer_index + 1
        act = graph.activations[layer_slot].reshape(-1)
        activations.append(act)
    acts = np.stack(activations)
    mean = np.mean(acts, axis=0)
    std = np.std(acts, axis=0)
    sparsity = np.mean(acts == 0.0, axis=0)
    neurons = [
        {
            "index": int(i),
            "mean": float(mean[i]),
            "std": float(std[i]),
            "sparsity": float(sparsity[i]),
        }
        for i in range(len(mean))
    ]
    return {"layer_index": layer_index, "neurons": neurons}
