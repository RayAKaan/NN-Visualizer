from __future__ import annotations

from typing import Any

import numpy as np


def _classify_region(rel_y: float, rel_x: float) -> str:
    v = "top" if rel_y < 0.33 else "bottom" if rel_y > 0.66 else "center"
    h = "left" if rel_x < 0.33 else "right" if rel_x > 0.66 else "center"
    if v == "center" and h == "center":
        return "center"
    return f"{v}-{h}"


class ExplanationBuilder:
    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.9:
            return "high"
        if confidence >= 0.7:
            return "medium"
        return "low"

    def build(
        self,
        activations: dict[str, list[float]],
        weights: dict[str, Any],
        prediction: int,
        probabilities: list[float],
        pixels: list[float] | None = None,
    ) -> dict[str, Any]:
        hidden2 = np.array(activations.get("hidden2", []), dtype=np.float32)
        w_h2_out = np.array(weights.get("hidden2_output", []), dtype=np.float32)
        probs = np.array(probabilities, dtype=np.float32)

        confidence = float(probs[prediction]) if probs.size and prediction >= 0 else 0.0
        level = self._confidence_level(confidence)

        if hidden2.size > 0 and w_h2_out.size > 0 and prediction < w_h2_out.shape[1]:
            contribs = hidden2 * w_h2_out[:, prediction]
            top_idx = np.argsort(np.abs(contribs))[-5:][::-1]
            top_neurons = [
                {
                    "layer": "hidden2",
                    "index": int(i),
                    "activation": float(hidden2[i]),
                    "contribution": float(contribs[i]),
                }
                for i in top_idx
            ]
        else:
            top_neurons = []

        evidence: list[str] = []
        if pixels and len(pixels) == 784:
            grid = np.array(pixels, dtype=np.float32).reshape(28, 28)
            quads = {
                "upper-left": float(np.mean(grid[:14, :14])),
                "upper-right": float(np.mean(grid[:14, 14:])),
                "lower-left": float(np.mean(grid[14:, :14])),
                "lower-right": float(np.mean(grid[14:, 14:])),
            }
            strongest_quad = max(quads, key=quads.get)
            evidence.append(f"Strongest stroke energy appears in the {strongest_quad} quadrant.")

        competitors = []
        if probs.size:
            top_digits = np.argsort(probs)[-4:][::-1]
            competitors = [
                {
                    "digit": int(d),
                    "probability": float(probs[d]),
                    "reason": "High competing probability compared to winner.",
                }
                for d in top_digits
                if int(d) != prediction
            ]

        uncertainty_notes: list[str] = []
        if confidence < 0.6:
            uncertainty_notes.append("Low confidence prediction; consider a cleaner stroke.")
        if len(competitors) and confidence - competitors[0]["probability"] < 0.15:
            uncertainty_notes.append("Top classes are close; model sees ambiguity between candidate digits.")

        if not evidence:
            evidence.append("Prediction driven by hidden-layer activation/contribution pattern.")

        return {
            "prediction": int(prediction),
            "confidence": confidence,
            "confidence_level": level,
            "top_neurons": top_neurons,
            "evidence": evidence,
            "competitors": competitors,
            "uncertainty_notes": uncertainty_notes,
        }

    def build_cnn(
        self,
        feature_maps: list[dict[str, Any]],
        dense_layers: dict[str, list[float]],
        prediction: int,
        probabilities: list[float],
    ) -> dict[str, Any]:
        confidence = float(max(probabilities)) if probabilities else 0.0
        confidence_level = (
            "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
        )

        active_filters = []
        for fm_info in feature_maps:
            if fm_info.get("layer_type") == "conv":
                for idx in fm_info.get("activation_ranking", [])[:5]:
                    mean_act = fm_info.get("mean_activations", [0])[idx]
                    active_filters.append(
                        {
                            "layer": fm_info.get("layer_name", ""),
                            "filter_index": idx,
                            "mean_activation": round(float(mean_act), 4),
                        }
                    )

        spatial_evidence = []
        for fm_info in feature_maps:
            if fm_info.get("layer_type") == "conv" and fm_info.get("activation_ranking"):
                top_filter = fm_info["activation_ranking"][0]
                fm = fm_info.get("feature_maps", {}).get(str(top_filter))
                if fm is None:
                    continue
                arr = np.array(fm)
                h, w = arr.shape
                total = arr.sum()
                if total > 0:
                    cy = np.sum(np.arange(h)[:, None] * arr) / total
                    cx = np.sum(np.arange(w)[None, :] * arr) / total
                    region = _classify_region(cy / h, cx / w)
                    spatial_evidence.append(
                        f"{fm_info['layer_name']}: strongest response in {region} region"
                    )

        sorted_probs = sorted(enumerate(probabilities), key=lambda x: -x[1])
        competitors = []
        for digit, prob in sorted_probs[1:4]:
            if prob > 0.05:
                competitors.append(
                    {
                        "digit": int(digit),
                        "probability": round(float(prob), 4),
                        "reason": f"probability {prob:.1%}",
                    }
                )

        uncertainty_notes = []
        if confidence < 0.7:
            uncertainty_notes.append("Low confidence — input may be ambiguous")
        if competitors and competitors[0]["probability"] > 0.2:
            uncertainty_notes.append(
                f"Strong competitor: digit {competitors[0]['digit']} at {competitors[0]['probability']:.1%}"
            )

        return {
            "model_type": "cnn",
            "prediction": int(prediction),
            "confidence": round(confidence, 4),
            "confidence_level": confidence_level,
            "active_filters": active_filters,
            "spatial_evidence": spatial_evidence,
            "competitors": competitors,
            "uncertainty_notes": uncertainty_notes,
            "dense_layers_summary": {k: len(v) for k, v in dense_layers.items()},
        }
