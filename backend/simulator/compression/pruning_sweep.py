from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from .magnitude_pruning import magnitude_prune


def pruning_sweep(graph: NetworkGraph, sparsity_range: List[float]) -> Dict:
    results = []
    for s in sparsity_range:
        res = magnitude_prune(graph, float(s))
        results.append({
            "sparsity": float(s),
            "accuracy": 0.0,
            "params": res["remaining_params"],
            "flops": 0,
        })
    return {"sweep_results": results, "recommended_sparsity": float(sparsity_range[len(sparsity_range)//2]) if sparsity_range else 0.5, "pareto_frontier": []}
