from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph


def magnitude_prune(graph: NetworkGraph, sparsity: float = 0.5) -> Dict:
    weights = [w.reshape(-1) for w in graph.weights]
    if not weights:
        return {
            "original_params": 0,
            "remaining_params": 0,
            "sparsity_achieved": 0.0,
            "per_layer_sparsity": [],
            "pruning_mask": [],
        }

    all_w = np.concatenate(weights)
    threshold = np.percentile(np.abs(all_w), sparsity * 100)
    masks = []
    per_layer = []
    remaining = 0
    total = 0
    for idx, w in enumerate(graph.weights):
        mask = (np.abs(w) > threshold).astype(np.float32)
        masks.append(mask)
        remaining += int(mask.sum())
        total += mask.size
        per_layer.append({
            "layer": idx,
            "sparsity": float(1.0 - mask.sum() / mask.size),
            "remaining": int(mask.sum()),
            "total": int(mask.size),
        })
    sparsity_achieved = 1.0 - (remaining / max(total, 1))
    return {
        "original_params": int(total),
        "remaining_params": int(remaining),
        "sparsity_achieved": float(sparsity_achieved),
        "per_layer_sparsity": per_layer,
        "pruning_mask": [m.tolist() for m in masks],
    }
