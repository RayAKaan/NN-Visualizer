"""PredictionService tests.

Light error-path tests always run; real inference tests are skipped unless the
caller opts in with NN_RUN_HEAVY=1 (they exercise genuine pretrained weights
and, on a fresh machine, trigger one-time lazy downloads).
"""
import base64
import os

import numpy as np
import pytest

from services.prediction_service import (
    ModelNotFoundError,
    ModelUnavailableError,
    prediction_service,
)

RUN_HEAVY = os.environ.get("NN_RUN_HEAVY", "") == "1"
heavy = pytest.mark.skipif(not RUN_HEAVY, reason="set NN_RUN_HEAVY=1 to run real model inference tests")

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(BACKEND_ROOT, "..", "frontend", "public", "samples", "grace_hopper.jpg")


def _digits():
    import tensorflow as tf

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    def pixels(digit):
        return (x_test[y_test == digit][0].astype("float32") / 255.0).reshape(-1).tolist()

    return pixels


def _std_keys(payload: dict) -> set:
    return set(payload.keys())


# ---------------------------------------------------------------------------
# Error paths (always run)
# ---------------------------------------------------------------------------
def test_predict_unknown_model():
    with pytest.raises(ModelNotFoundError):
        prediction_service.predict({"model_id": "does-not-exist", "pixels": [0.0] * 784})


def test_predict_unavailable_model_lists_reason():
    with pytest.raises(ModelUnavailableError) as exc:
        prediction_service.predict({"model_id": "rnn-jongador-lstm-imdb-256", "pixels": [0.0] * 784})
    assert "unavailable" in str(exc.value)


def test_load_unavailable_model_raises():
    with pytest.raises(ModelUnavailableError):
        prediction_service.load("rnn-pratyushee-assamese-lstm")


def test_unload_unknown_model():
    with pytest.raises(ModelNotFoundError):
        prediction_service.unload("nope")


def test_legacy_model_type_resolves_to_baseline():
    model_id = prediction_service._legacy_model_type_to_id("ann")
    assert model_id == "ann-nn-visualizer"


# ---------------------------------------------------------------------------
# Real inference (NN_RUN_HEAVY=1)
# ---------------------------------------------------------------------------
@heavy
def test_legacy_ann_standardized_payload_and_digit():
    r = prediction_service.predict({"model_id": "ann-nn-visualizer", "pixels": _digits()(7)})
    assert r["predicted_class"] == "7"
    assert r["predicted_index"] == 7
    assert r["confidence"] > 0.9
    assert len(r["probabilities"]) == 10
    assert r["labels"] == [str(i) for i in range(10)]
    expected = {
        "model_id", "model_name", "family", "framework", "predicted_class",
        "predicted_index", "confidence", "probabilities", "labels", "top_k",
        "latency_ms", "device", "input_shape", "architecture", "parameter_count",
        "dataset", "preprocessing",
    }
    assert expected.issubset(_std_keys(r))
    assert r["family"] == "ANN"
    assert r["device"] in ("CPU", "GPU")
    # legacy per-layer trace survives for the baseline model
    assert isinstance(r.get("details", {}).get("layers", {}).get("hidden1"), list)


@heavy
def test_repeated_inference_is_stable_and_cached():
    px = _digits()(7)
    first = prediction_service.predict({"model_id": "ann-nn-visualizer", "pixels": px})
    second = prediction_service.predict({"model_id": "ann-nn-visualizer", "pixels": px})
    assert first["predicted_index"] == second["predicted_index"] == 7
    assert abs(first["confidence"] - second["confidence"]) < 1e-6


@heavy
def test_dacorvo_ann_digit():
    r = prediction_service.predict({"model_id": "ann-dacorvo-mnist-mlp", "pixels": _digits()(4)})
    assert r["predicted_index"] == 4
    assert r["confidence"] > 0.9


@heavy
def test_mysticdan_ann_digit():
    r = prediction_service.predict({"model_id": "ann-mysticdan-mlp-mnist", "pixels": _digits()(9)})
    assert r["predicted_index"] == 9
    assert r["confidence"] > 0.9


@heavy
def test_rnn_kerasio_negative_and_positive():
    neg = prediction_service.predict({
        "model_id": "rnn-bilstm-imdb-kerasio",
        "text": "This movie was an absolute disaster, boring and badly acted.",
    })
    pos = prediction_service.predict({
        "model_id": "rnn-bilstm-imdb-kerasio",
        "text": "A wonderful, clever film with superb acting and a touching story.",
    })
    assert neg["predicted_class"] == "negative"
    assert neg["confidence"] > 0.8
    assert pos["predicted_class"] == "positive"
    assert pos["confidence"] > 0.8
    assert neg["labels"] == ["negative", "positive"]


@heavy
def test_cnn_mobilenetv3small_imagenet_class_names():
    if not os.path.exists(SAMPLE):
        pytest.skip("sample image missing")
    b64 = base64.b64encode(open(SAMPLE, "rb").read()).decode()
    r = prediction_service.predict({"model_id": "cnn-mobilenetv3small", "image": b64})
    assert r["predicted_class"] == "military_uniform"
    assert r["predicted_index"] == 652
    assert r["confidence"] > 0.5
    assert len(r["probabilities"]) == 1000
    top = r["top_k"][0]
    assert top["label"] == "military_uniform"
    # all class labels are real ImageNet names, not raw indices
    assert all(isinstance(lbl, str) and not lbl.isdigit() for lbl in r["labels"][:50])


@heavy
def test_models_can_be_unloaded_after_predict():
    prediction_service.load("ann-dacorvo-mnist-mlp")
    assert "ann-dacorvo-mnist-mlp" in prediction_service.registry.loaded_ids()
    prediction_service.unload("ann-dacorvo-mnist-mlp")
    assert "ann-dacorvo-mnist-mlp" not in prediction_service.registry.loaded_ids()
