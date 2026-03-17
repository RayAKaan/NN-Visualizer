from __future__ import annotations

from typing import Dict, List
import numpy as np

from .integrated_gradients import integrated_gradients
from .deep_shap import deep_shap
from .lime_explainer import lime_explain
from .lrp import lrp_explain


def _flatten(arr: List[float]) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    ar = a.argsort().argsort().astype(np.float32)
    br = b.argsort().argsort().astype(np.float32)
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def compare_methods(graph, input_vec: List[float], target_class: int) -> Dict:
    ig = integrated_gradients(graph, input_vec, target_class, 50)
    shap = deep_shap(graph, input_vec, target_class, 30)
    lrp = lrp_explain(graph, input_vec, target_class, 1e-2)
    lime = lime_explain(graph, input_vec, None, target_class, 100)

    ig_flat = _flatten(ig.get("attributions", []))
    shap_flat = _flatten(shap.get("shap_values", []))
    lrp_flat = _flatten(lrp.get("input_relevance", []))

    return {
        "per_method": {
            "integrated_gradients": {"image_base64": ig.get("attributions_base64")},
            "shap": {"image_base64": shap.get("shap_base64")},
            "lrp": {"image_base64": lrp.get("input_relevance_base64")},
            "lime": {"image_base64": lime.get("explanation_image_base64")},
        },
        "agreement": {
            "pairwise_correlation": {
                "ig_shap": _spearman(ig_flat, shap_flat),
                "ig_lrp": _spearman(ig_flat, lrp_flat),
                "shap_lrp": _spearman(shap_flat, lrp_flat),
            },
            "top_10_overlap": {},
            "full_consensus_ratio": 0.0,
        },
        "recommendation": "Integrated Gradients and LRP agreement computed.",
    }
