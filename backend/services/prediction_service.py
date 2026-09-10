"""Model-agnostic prediction service used by the Prediction API."""
from __future__ import annotations

import threading
import time
from typing import Any

from model_registry.base import ModelLoadError
from model_registry.registry import ModelRegistry, registry
from model_registry.schema import prediction_payload


class ModelNotFoundError(LookupError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


def detect_device() -> str:
    """Report the device TensorFlow would execute on.

    CPU is the guaranteed baseline; no CUDA/GPU is required.
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "GPU"
        return "CPU"
    except Exception:
        return "CPU"


class PredictionService:
    """Resolves ``model_id`` -> adapter -> lazy load -> predict."""

    def __init__(self, model_registry: ModelRegistry):
        self.registry = model_registry
        self._predict_lock = threading.Lock()

    # ------------------------------------------------------------------
    def predict(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        model_id = raw_input.get("model_id")
        if not model_id:
            model_id = self._legacy_model_type_to_id(raw_input.get("model_type"))
        if not model_id:
            raise ModelNotFoundError(
                "Missing 'model_id' (or legacy 'model_type'). See GET /models/catalog."
            )
        reason = self.registry.unavailable_reason(model_id)
        if reason:
            raise ModelUnavailableError(f"{model_id} is unavailable: {reason}")
        if not self.registry.contains(model_id):
            raise ModelNotFoundError(f"Unknown model_id: {model_id}")

        adapter = self.registry.get(model_id)
        started = time.time()
        device = detect_device()
        try:
            adapter.ensure_loaded()
        except ModelLoadError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        except Exception as exc:  # surface download/load failures cleanly
            raise ModelUnavailableError(f"{model_id} failed to load: {exc}") from exc

        try:
            idx, conf, probs, labels, extra = adapter.predict(raw_input)
        except ModelLoadError as exc:
            raise ModelUnavailableError(str(exc)) from exc
        latency_ms = (time.time() - started) * 1000.0
        return prediction_payload(
            adapter,
            predicted_index=idx,
            confidence=conf,
            probabilities=probs,
            labels=labels,
            latency_ms=round(latency_ms, 1),
            device=device,
            extra=extra,
        )

    # ------------------------------------------------------------------
    def load(self, model_id: str) -> dict[str, Any]:
        reason = self.registry.unavailable_reason(model_id)
        if reason:
            raise ModelUnavailableError(f"{model_id} is unavailable: {reason}")
        if not self.registry.contains(model_id):
            raise ModelNotFoundError(f"Unknown model_id: {model_id}")
        started = time.time()
        try:
            self.registry.load(model_id)
        except Exception as exc:
            raise ModelUnavailableError(f"{model_id} failed to load: {exc}") from exc
        return {"model_id": model_id, "loaded": True, "load_ms": round((time.time() - started) * 1000.0, 1)}

    def unload(self, model_id: str) -> dict[str, Any]:
        reason = self.registry.unavailable_reason(model_id)
        if reason:
            raise ModelUnavailableError(f"{model_id} is unavailable: {reason}")
        if not self.registry.contains(model_id):
            raise ModelNotFoundError(f"Unknown model_id: {model_id}")
        self.registry.unload(model_id)
        return {"model_id": model_id, "loaded": False}

    @staticmethod
    def _legacy_model_type_to_id(model_type):
        return {
            "ann": "ann-nn-visualizer",
            "cnn": "cnn-nn-visualizer",
            "rnn": "rnn-nn-visualizer",
        }.get((model_type or "").lower())


prediction_service = PredictionService(registry)
