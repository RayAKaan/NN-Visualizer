from __future__ import annotations

from typing import Dict, List
import numpy as np

from ..graph_engine import NetworkGraph
from .pgd import pgd_attack


def evaluate_robustness(
    graph: NetworkGraph,
    samples: List[tuple[np.ndarray, int]],
    epsilons: List[float],
    steps: int = 5,
    step_size: float | None = None,
) -> Dict:
    results = []
    for eps in epsilons:
        correct = 0
        total = max(len(samples), 1)
        for x, label in samples:
            adv = pgd_attack(
                graph,
                x.tolist(),
                int(label),
                float(eps),
                float(step_size or (eps / 4.0 + 1e-6)),
                steps,
            )
            pred = adv["adversarial_prediction"]["class"]
            if pred == int(label):
                correct += 1
        results.append({"epsilon": float(eps), "accuracy": correct / total, "mean_confidence": 0.0})
    return {"robustness_curve": results}
