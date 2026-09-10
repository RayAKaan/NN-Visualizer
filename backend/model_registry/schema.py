"""Shared metadata / response schema for the pretrained model registry."""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Model metadata keys (mirrors the "MODEL METADATA" contract in the spec).
# ---------------------------------------------------------------------------
MODEL_META_FIELDS = [
    "id",
    "name",
    "family",            # "ANN" | "CNN" | "RNN"
    "framework",
    "source",
    "weights",
    "dataset",
    "input_type",        # "mnist_pixels" | "image" | "text"
    "input_shape",
    "num_classes",
    "preprocessing",
    "architecture",
    "parameter_count",
    "description",
    "license",
    "pretrained",
]

# statuses the registry reports
STATUS_AVAILABLE = "available"      # registered + runnable (may still need a lazy download)
STATUS_LOADING = "loading"
STATUS_LOADED = "loaded"
STATUS_ERROR = "error"
STATUS_UNAVAILABLE = "unavailable"  # intentionally listed but not runnable here

# family names as exposed to the UI / metadata
FAMILY_ANN = "ANN"
FAMILY_CNN = "CNN"
FAMILY_RNN = "RNN"
FAMILIES = [FAMILY_ANN, FAMILY_CNN, FAMILY_RNN]

FAMILY_KEYS = {"ann": FAMILY_ANN, "cnn": FAMILY_CNN, "rnn": FAMILY_RNN}


def family_display(key: str) -> str:
    """Map a lowercase family key ('ann') to its display name ('ANN')."""
    return FAMILY_KEYS.get(str(key).lower(), str(key).upper())


def _build_metadata(**kwargs: Any) -> dict[str, Any]:
    meta = {
        "framework": None,
        "source": None,
        "weights": None,
        "dataset": None,
        "input_type": None,
        "input_shape": None,
        "num_classes": None,
        "preprocessing": None,
        "architecture": None,
        "parameter_count": None,
        "description": "",
        "license": None,
        "pretrained": True,
    }
    meta.update({k: v for k, v in kwargs.items() if v is not None})
    return meta


# ---------------------------------------------------------------------------
# Standardized prediction response helpers.
# ---------------------------------------------------------------------------
def prediction_payload(adapter, *, predicted_index, confidence, probabilities,
                       labels=None, latency_ms=None, device="CPU", extra=None):
    """Build the standardized prediction object returned by /predict.

    ``probabilities`` is a list of floats aligned to ``labels`` (or class
    indices when ``labels`` is None).  ``extra`` may carry model-specific
    visualization payloads (e.g. legacy MNIST activations) which the UI can
    optionally render; it is never required by the standardized schema.
    """
    labels = labels or [str(i) for i in range(len(probabilities))]
    probs = [float(p) for p in probabilities]
    top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top_k = [
        {"index": int(i), "label": str(labels[i]), "probability": probs[i]}
        for i in top[:5]
    ]
    predicted_class = str(labels[predicted_index])

    payload: dict[str, Any] = {
        "model_id": adapter.model_id,
        "model_name": adapter.metadata["name"],
        "family": adapter.metadata["family"],
        "framework": adapter.metadata.get("framework"),
        "predicted_class": predicted_class,
        "predicted_index": int(predicted_index),
        "confidence": float(confidence),
        "probabilities": probs,
        "labels": list(labels),
        "top_k": top_k,
        "latency_ms": latency_ms,
        "device": device,
        "input_shape": adapter.metadata.get("input_shape"),
        "architecture": adapter.metadata.get("architecture"),
        "parameter_count": adapter.metadata.get("parameter_count"),
        "dataset": adapter.metadata.get("dataset"),
        "preprocessing": adapter.metadata.get("preprocessing"),
    }
    if extra:
        payload["details"] = extra
    return payload
