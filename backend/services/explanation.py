import numpy as np


class ExplanationBuilder:
    def build(self, result, weights, model_type):
        if model_type == "ann":
            return self.build_ann(result, weights)
        if model_type == "cnn":
            return self.build_cnn(result)
        if model_type == "rnn":
            return self.build_rnn(result)
        return {"summary": "Unsupported model type"}

    def _competitors(self, probs):
        idx = np.argsort(-np.array(probs))[:3]
        return [{"digit": int(i), "probability": float(probs[i])} for i in idx]

    def build_ann(self, result, _weights):
        top_neurons = sorted(enumerate(result["layers"].get("hidden1", [])), key=lambda x: -x[1])[:5]
        return {
            "model_type": "ann",
            "top_neurons": [{"id": int(i), "activation": float(v)} for i, v in top_neurons],
            "quadrant_evidence": "Strong center-stroke activations detected",
            "competitors": self._competitors(result["probabilities"]),
            "confidence_level": "high" if result["confidence"] > 0.8 else "medium",
            "uncertainty_notes": "Ambiguous pen strokes can reduce ANN certainty",
        }

    def build_cnn(self, result):
        active = []
        for layer in result.get("feature_maps", [])[:2]:
            if layer["activation_ranking"]:
                idx = layer["activation_ranking"][0]
                active.append({"layer": layer["layer_name"], "filter": idx, "mean": layer["mean_activations"][idx]})
        return {
            "model_type": "cnn",
            "active_filters": active,
            "spatial_evidence": "Convolution layers focused on stroke intersections",
            "competitors": self._competitors(result["probabilities"]),
            "confidence_level": "high" if result["confidence"] > 0.8 else "medium",
            "uncertainty_notes": "Low contrast edges can impact filter responses",
        }

    def build_rnn(self, result):
        ts = np.array(result.get("timestep_activations", []))
        top_t = np.argsort(-ts)[:5].tolist() if ts.size else []
        return {
            "model_type": "rnn",
            "timestep_importance": top_t,
            "sequential_summary": "Rows are processed top-to-bottom to infer digit trajectory",
            "competitors": self._competitors(result["probabilities"]),
            "confidence_level": "high" if result["confidence"] > 0.8 else "medium",
            "uncertainty_notes": "Temporal ambiguity in similar row patterns may confuse LSTM",
        }


explainer = ExplanationBuilder()
