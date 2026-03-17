from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..graph_engine import NetworkGraph
from .rendering import render_heatmap_overlay, resize_nearest


def compute_grad_cam(graph: NetworkGraph, input_vec: List[float], input_shape: List[int], target_class: int) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32).reshape(input_shape)
    graph.forward(x)
    output = graph.activations[-1].reshape(-1)
    if target_class < 0 or target_class >= len(output):
        target_class = int(np.argmax(output))
    d_out = np.zeros_like(output)
    d_out[target_class] = 1.0

    last_conv_idx = -1
    for idx in range(1, len(graph.layer_instances)):
        if graph.layers[idx].layer_type == "conv2d":
            last_conv_idx = idx
    if last_conv_idx < 0:
        raise ValueError("No conv2d layer found for Grad-CAM")

    grad = d_out
    for idx in reversed(range(1, len(graph.layer_instances))):
        layer = graph.layer_instances[idx]
        grad = layer.backward(grad)
        if idx == last_conv_idx:
            break

    conv_out = graph.activations[last_conv_idx]
    if grad.ndim == 1:
        grad = grad.reshape(conv_out.shape)
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(conv_out.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_out[i]
    cam = np.maximum(cam, 0.0)
    cam = cam / (np.max(cam) + 1e-8)
    base = x.reshape(input_shape)[-1] if x.ndim == 3 else x.reshape(input_shape)
    cam_up = resize_nearest(cam, base.shape)
    base64_input, overlay = render_heatmap_overlay(base, cam_up)
    return {
        "method": "grad_cam",
        "saliency_map": cam_up.tolist(),
        "saliency_base64": base64_input,
        "overlay_base64": overlay,
        "top_pixels": [],
    }
