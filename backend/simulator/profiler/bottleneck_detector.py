from __future__ import annotations

from typing import Dict, List


def detect_bottlenecks(per_layer: List[Dict], memory_per_layer: List[Dict]) -> Dict:
    compute_bottleneck = None
    memory_bottleneck = None

    if per_layer:
        max_flops = max(per_layer, key=lambda x: x.get("flops_forward", 0))
        compute_bottleneck = {
            "layer": max_flops.get("layer_index"),
            "type": max_flops.get("layer_type"),
            "reason": f"{max_flops.get('flops_forward', 0)} FLOPs",
        }

    if memory_per_layer:
        max_mem = max(memory_per_layer, key=lambda x: x.get("activation_bytes", 0))
        memory_bottleneck = {
            "layer": max_mem.get("layer_index"),
            "type": "activation",
            "reason": f"{max_mem.get('activation_bytes', 0)} bytes",
        }

    return {
        "compute_bottleneck": compute_bottleneck,
        "memory_bottleneck": memory_bottleneck,
    }
