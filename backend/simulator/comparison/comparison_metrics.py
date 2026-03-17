from __future__ import annotations

from typing import Dict, List

from .comparison_engine import comparison_manager


def _pick_winner(models: List[Dict]) -> Dict | None:
    if not models:
        return None

    def score(m: Dict) -> tuple:
        return (
            float(m.get("final_test_accuracy", 0.0)),
            -float(m.get("final_test_loss", 0.0)),
            -float(m.get("final_train_accuracy", 0.0)),
        )

    best = max(models, key=score)
    return {
        "overall": best.get("model_id"),
        "per_metric": {
            "accuracy": best.get("model_id"),
            "loss": best.get("model_id"),
            "params": min(models, key=lambda m: m.get("total_params", 0)).get("model_id"),
        },
    }


def compute_comparison_results(comparison_id: str) -> Dict:
    results = comparison_manager.get(comparison_id)
    models = results.get("models", [])
    if models and not results.get("winner"):
        results["winner"] = _pick_winner(models)
    return results
