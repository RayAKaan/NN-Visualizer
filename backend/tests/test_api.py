"""HTTP-level tests for the Prediction API (catalog, load/unload, predict)."""
import base64
import os

import pytest

RUN_HEAVY = os.environ.get("NN_RUN_HEAVY", "") == "1"
heavy = pytest.mark.skipif(not RUN_HEAVY, reason="set NN_RUN_HEAVY=1 to run real model inference tests")

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(BACKEND_ROOT, "..", "frontend", "public", "samples", "grace_hopper.jpg")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app import app

    with TestClient(app) as c:  # runs lifespan (loads legacy weights)
        yield c


# ---------------------------------------------------------------------------
# Catalog + status (always run)
# ---------------------------------------------------------------------------
def test_catalog_shape(client):
    res = client.get("/models/catalog")
    assert res.status_code == 200
    data = res.json()
    assert set(data) == {"models", "families", "loaded"}
    assert data["families"] == ["ANN", "CNN", "RNN"]
    ids = [m["id"] for m in data["models"]]
    assert "ann-dacorvo-mnist-mlp" in ids
    assert "cnn-mobilenetv3small" in ids
    assert "rnn-bilstm-imdb-kerasio" in ids
    # intentionally-unavailable models are surfaced with reasons
    by_id = {m["id"]: m for m in data["models"]}
    assert by_id["rnn-jongador-lstm-imdb-256"]["status"] == "unavailable"
    assert by_id["rnn-jongador-lstm-imdb-256"]["unavailable_reason"]


def test_catalog_single_entry(client):
    res = client.get("/models/catalog/cnn-vgg16")
    assert res.status_code == 200
    m = res.json()
    assert m["id"] == "cnn-vgg16"
    assert m["input_type"] == "image"
    assert m["num_classes"] == 1000


def test_catalog_unknown_entry_is_404(client):
    assert client.get("/models/catalog/nope").status_code == 404


def test_load_unavailable_model_returns_503(client):
    res = client.post("/models/rnn-jongador-lstm-imdb-256/load")
    assert res.status_code == 503
    assert "unavailable" in res.json()["detail"]


def test_load_unload_unknown_model_404(client):
    assert client.post("/models/nope/load").status_code == 404
    assert client.post("/models/nope/unload").status_code == 404


def test_predict_requires_some_identifier(client):
    res = client.post("/predict", json={})
    assert res.status_code == 404


def test_predict_unknown_model_404(client):
    res = client.post("/predict", json={"model_id": "ghost", "pixels": [0.0] * 784})
    assert res.status_code == 404


def test_predict_ann_missing_pixels_is_400(client):
    res = client.post("/predict", json={"model_id": "ann-dacorvo-mnist-mlp"})
    assert res.status_code == 400


def test_legacy_model_type_cnn_clean_503(client):
    res = client.post("/predict", json={"model_type": "cnn", "pixels": [0.0] * 784})
    assert res.status_code == 503
    assert "weights are not loaded" in res.json()["detail"]


# ---------------------------------------------------------------------------
# Real predictions through the HTTP layer (NN_RUN_HEAVY=1)
# ---------------------------------------------------------------------------
def _mnist_digit(d):
    import tensorflow as tf

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_test[y_test == d][0].astype("float32") / 255.0).reshape(-1).tolist()


@heavy
def test_http_predict_legacy_ann_layers(client):
    res = client.post("/predict", json={"model_type": "ann", "pixels": _mnist_digit(7)})
    assert res.status_code == 200
    body = res.json()
    assert body["prediction"] == 7
    assert body["model_type"] == "ann"
    assert body["layers"]["hidden1"]
    assert body["explanation"] is not None


@heavy
def test_http_predict_registry_ann(client):
    res = client.post("/predict", json={"model_id": "ann-nn-visualizer", "pixels": _mnist_digit(7)})
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] == "7"
    assert body["family"] == "ANN"
    assert "details" in body and isinstance(body["details"]["layers"].get("hidden1"), list)


@heavy
def test_http_predict_rnn(client):
    res = client.post(
        "/predict",
        json={
            "model_id": "rnn-bilstm-imdb-kerasio",
            "text": "This movie was an absolute disaster, boring and badly acted.",
        },
    )
    assert res.status_code == 200
    assert res.json()["predicted_class"] == "negative"


@heavy
def test_http_predict_cnn(client):
    if not os.path.exists(SAMPLE):
        pytest.skip("sample image missing")
    b64 = base64.b64encode(open(SAMPLE, "rb").read()).decode()
    res = client.post("/predict", json={"model_id": "cnn-mobilenetv3small", "image": b64})
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] == "military_uniform"
    assert body["confidence"] > 0.5


@heavy
def test_http_load_then_catalog_shows_loaded(client):
    client.post("/models/ann-dacorvo-mnist-mlp/load")
    catalog = client.get("/models/catalog").json()
    by_id = {m["id"]: m for m in catalog["models"]}
    assert by_id["ann-dacorvo-mnist-mlp"]["status"] == "loaded"
    assert "ann-dacorvo-mnist-mlp" in catalog["loaded"]
    client.post("/models/ann-dacorvo-mnist-mlp/unload")
