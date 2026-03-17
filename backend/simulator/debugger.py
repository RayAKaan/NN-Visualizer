from __future__ import annotations

from typing import Dict, List

import numpy as np


def diagnose(
    gradient_norms: List[float],
    loss_history: List[float],
    train_loss: float | None,
    test_loss: float | None,
    dead_neurons: List[int],
    total_neurons: List[int],
) -> Dict:
    issues = []
    overall = "healthy"

    for idx, g in enumerate(gradient_norms):
        if g < 1e-7:
            issues.append(
                {
                    "code": "VANISHING_GRADIENT",
                    "severity": "critical",
                    "layer_index": idx,
                    "message": f"Layer {idx}: Vanishing gradients (norm={g:.2e})",
                    "suggestion": "Try LeakyReLU or increase learning rate",
                }
            )
            overall = "warning"
        elif g < 1e-4:
            issues.append(
                {
                    "code": "VANISHING_GRADIENT",
                    "severity": "warning",
                    "layer_index": idx,
                    "message": f"Layer {idx}: small gradients (norm={g:.2e})",
                    "suggestion": "Try LeakyReLU or increase learning rate",
                }
            )
            overall = "warning"
        if g > 100:
            issues.append(
                {
                    "code": "EXPLODING_GRADIENT",
                    "severity": "critical",
                    "layer_index": idx,
                    "message": f"Layer {idx}: exploding gradients (norm={g:.2e})",
                    "suggestion": "Reduce learning rate or add clipping",
                }
            )
            overall = "warning"
        elif g > 10:
            issues.append(
                {
                    "code": "EXPLODING_GRADIENT",
                    "severity": "warning",
                    "layer_index": idx,
                    "message": f"Layer {idx}: large gradients (norm={g:.2e})",
                    "suggestion": "Reduce learning rate or add clipping",
                }
            )
            overall = "warning"

    for idx, dead in enumerate(dead_neurons):
        total = total_neurons[idx] if idx < len(total_neurons) else 0
        if total == 0:
            continue
        ratio = dead / max(total, 1)
        if ratio > 0.9:
            issues.append(
                {
                    "code": "DEAD_NEURON",
                    "severity": "critical",
                    "layer_index": idx,
                    "neuron_indices": [],
                    "message": f"Layer {idx}: {dead}/{total} neurons dead",
                    "suggestion": "Use LeakyReLU or reinitialize weights",
                }
            )
            overall = "warning"
        elif ratio > 0.5:
            issues.append(
                {
                    "code": "DEAD_NEURON",
                    "severity": "warning",
                    "layer_index": idx,
                    "neuron_indices": [],
                    "message": f"Layer {idx}: {dead}/{total} neurons dead",
                    "suggestion": "Use LeakyReLU or reduce learning rate",
                }
            )
            overall = "warning"

    if train_loss is not None and test_loss is not None:
        if test_loss > train_loss * 1.5:
            issues.append(
                {
                    "code": "OVERFITTING",
                    "severity": "warning",
                    "layer_index": None,
                    "message": "Overfitting suspected (test loss >> train loss)",
                    "suggestion": "Add dropout or increase L2",
                }
            )
            overall = "warning"

    loss_health = {
        "trend": "decreasing",
        "overfitting_score": 0.0,
        "is_overfitting": False,
        "is_diverging": False,
        "is_plateau": False,
    }
    if len(loss_history) >= 10:
        recent = np.array(loss_history[-10:])
        loss_health["trend"] = "decreasing" if recent[-1] < recent[0] else "increasing"

    return {
        "overall_health": overall,
        "issues": issues,
        "gradient_flow": {
            "per_layer_norms": gradient_norms,
            "flow_ratio": gradient_norms[-1] / max(gradient_norms[0], 1e-8) if gradient_norms else 0.0,
            "health": "healthy" if overall == "healthy" else "warning",
        },
        "neuron_health": {
            "per_layer_alive": [t - d for t, d in zip(total_neurons, dead_neurons)],
            "per_layer_total": total_neurons,
            "dead_neuron_map": {},
        },
        "loss_health": loss_health,
    }

