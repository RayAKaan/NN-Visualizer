from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..graph_engine import NetworkGraph
from .rendering import render_gray


def compute_feature_maps(graph: NetworkGraph, input_image: List[float], input_shape: List[int]) -> Dict:
    x = np.asarray(input_image, dtype=np.float32).reshape(input_shape)
    graph.forward(x)
    per_layer = []
    for idx in range(1, len(graph.layer_instances)):
        layer = graph.layer_instances[idx]
        layer_type = graph.layers[idx].layer_type
        if layer_type not in {"conv2d", "maxpool2d", "avgpool2d"}:
            continue
        out = graph.activations[idx]
        maps = []
        for f in range(out.shape[0]):
            fmap = out[f]
            maps.append(
                {
                    "filter_index": int(f),
                    "map": fmap.tolist(),
                    "map_base64": render_gray(fmap),
                    "max_activation": float(np.max(fmap)),
                    "mean_activation": float(np.mean(fmap)),
                    "sparsity": float(np.mean(fmap == 0.0)),
                }
            )
        kernels = []
        if layer_type == "conv2d" and hasattr(layer, "K"):
            for f in range(layer.K.shape[0]):
                kernels.append(
                    {
                        "filter_index": int(f),
                        "kernel": layer.K[f, 0].tolist() if layer.K.shape[1] > 0 else layer.K[f].tolist(),
                        "kernel_base64": render_gray(layer.K[f, 0] if layer.K.shape[1] > 0 else layer.K[f]),
                        "description": "kernel",
                    }
                )
        per_layer.append(
            {
                "layer_index": idx - 1,
                "layer_type": layer_type,
                "n_filters": int(out.shape[0]),
                "output_shape": [int(s) for s in out.shape],
                "feature_maps": maps,
                "filter_kernels": kernels,
            }
        )
    return {"per_layer": per_layer}
