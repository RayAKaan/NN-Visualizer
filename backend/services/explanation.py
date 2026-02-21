from __future__ import annotations

from typing import Any

import numpy as np


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
        weights: dict[str, list[list[float]]],
        prediction: int,
        probabilities: list[float],
        pixels: list[float] | None = None,
    ) -> dict[str, Any]:
        hidden2 = np.array(activations.get("hidden2", []), dtype=np.float32)
        w_h2_out = np.array(weights.get("hidden2_output", []), dtype=np.float32)
        probs = np.array(probabilities, dtype=np.float32)

        confidence = float(probs[prediction]) if probs.size else 0.0
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

        if probs.size:
            top_digits = np.argsort(probs)[-3:][::-1]
            competitors = [
                {
                    "digit": int(d),
                    "probability": float(probs[d]),
                    "reason": "High competing probability compared to winner.",
                }
                for d in top_digits
                if int(d) != prediction
            ]
        else:
            competitors = []

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
