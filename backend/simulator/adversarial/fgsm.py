from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..backward_engine import backward_full
from ..graph_engine import NetworkGraph
from ..loss_functions import compute_loss
from ..visualization.rendering import render_heatmap_overlay


def fgsm_attack(
    graph: NetworkGraph,
    input_vec: List[float],
    true_label: int,
    epsilon: float,
    input_shape: List[int] | None = None,
) -> Dict:
    x = np.asarray(input_vec, dtype=np.float32)
    y = graph.forward(x)
    target = np.zeros_like(y)
    if target.size > true_label:
        target[true_label] = 1.0

    back = backward_full(graph, x.tolist(), target.tolist(), "mse", 0.0)
    grad = np.asarray(back.get("input_gradient", np.zeros_like(x)), dtype=np.float32)
    adv = x + epsilon * np.sign(grad)

    y_adv = graph.forward(adv)
    orig_pred = int(np.argmax(y)) if y.size else 0
    adv_pred = int(np.argmax(y_adv)) if y_adv.size else 0

    linf = float(np.max(np.abs(adv - x))) if adv.size else 0.0
    l2 = float(np.linalg.norm(adv - x)) if adv.size else 0.0
    l0 = int(np.sum(np.abs(adv - x) > 1e-8)) if adv.size else 0

    base64_img = ""
    perturb_img = ""
    if input_shape and len(input_shape) >= 2:
        h, w = input_shape[-2], input_shape[-1]
        if h * w == x.size:
            base = x.reshape(h, w)
            heat = (adv - x).reshape(h, w)
            base64_img, perturb_img = render_heatmap_overlay(base, np.abs(heat), 0.5)

    return {
        "adversarial_input": adv.tolist(),
        "perturbation": (adv - x).tolist(),
        "adversarial_image_base64": base64_img,
        "perturbation_amplified_base64": perturb_img,
        "original_prediction": {"class": orig_pred, "confidence": float(y[orig_pred]) if y.size else 0.0},
        "adversarial_prediction": {"class": adv_pred, "confidence": float(y_adv[adv_pred]) if y_adv.size else 0.0},
        "attack_success": orig_pred != adv_pred,
        "perturbation_stats": {
            "linf_norm": linf,
            "l2_norm": l2,
            "l0_norm": l0,
            "mean_perturbation": float(np.mean(np.abs(adv - x))) if adv.size else 0.0,
        },
        "confidence_shift": [
            {"class": i, "original": float(y[i]), "adversarial": float(y_adv[i])} for i in range(len(y))
        ],
    }
