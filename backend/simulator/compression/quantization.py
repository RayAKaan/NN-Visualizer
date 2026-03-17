from __future__ import annotations

from typing import Dict
import numpy as np

from ..graph_engine import NetworkGraph


def quantize_weights(graph: NetworkGraph, target_dtype: str = "int8") -> Dict:
    target = (target_dtype or "int8").lower()
    results = []
    original_bytes = 0
    quant_bytes = 0
    for idx, w in enumerate(graph.weights):
        original_bytes += w.size * 4
        if target == "fp16":
            q = w.astype(np.float16)
            approx = q.astype(np.float32)
            quant_bytes += w.size * 2
        else:
            w_min = float(w.min())
            w_max = float(w.max())
            scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
            zero_point = int(round(-w_min / scale)) if scale != 0 else 0
            q = np.clip(np.round(w / scale) + zero_point, 0, 255).astype(np.uint8)
            approx = (q.astype(np.float32) - zero_point) * scale
            quant_bytes += w.size
        error = np.abs(approx - w)
        results.append({
            "layer": idx,
            "mean_error": float(error.mean()),
            "max_error": float(error.max()),
        })
    ratio = (original_bytes / max(quant_bytes, 1)) if original_bytes else 1.0
    return {
        "original_memory_bytes": int(original_bytes),
        "quantized_memory_bytes": int(quant_bytes),
        "compression_ratio": float(ratio),
        "per_layer_quantization_error": results,
    }
