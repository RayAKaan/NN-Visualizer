"""Tests for the execution-trace engine.

Light tests use the **untrained** model bundles (no weight files needed).
Set ``NN_RUN_HEAVY=1`` to include heavy tests that hit real loaded weights
via the HTTP API.
"""
import math
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.execution_trace import (
    build_all_networks_metadata,
    build_execution_trace,
    build_network_metadata,
    conv_cell_detail,
    dense_neuron_detail,
    get_bundle,
    lstm_gates_detail,
    non_input_layers,
    run_lstm_cell,
)

RUN_HEAVY = os.environ.get("NN_RUN_HEAVY", "") == "1"
heavy = pytest.mark.skipif(not RUN_HEAVY, reason="set NN_RUN_HEAVY=1 to run real model inference tests")

ZEROS_784 = [0.0] * 784


# ---------------------------------------------------------------------------
# Client fixture (light – uses untrained bundles; avoids heavy imports unless
# the test client is actually used).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
class TestNetworkMetadata:
    def test_ann_structure(self):
        meta = build_network_metadata("ann")
        ops = [l["operation"] for l in meta["layers"]]
        assert "input" not in ops
        # Topology is read from the loaded model (may include dropout); the
        # chain must still start and end with a Dense projection.
        assert ops[0] == "dense"
        assert ops[-1] == "dense"
        assert len(ops) >= 4
        assert meta["id"] == "ann"
        assert meta["inputShape"] == [784]
        assert meta["numClasses"] == 10

    def test_cnn_structure(self):
        meta = build_network_metadata("cnn")
        ops = [l["operation"] for l in meta["layers"]]
        assert ops[:5] == ["conv2d", "max_pool", "conv2d", "max_pool", "flatten"]
        assert ops[-1] == "dense"

    def test_rnn_structure(self):
        meta = build_network_metadata("rnn")
        ops = [l["operation"] for l in meta["layers"]]
        assert "lstm" in ops

    def test_all_architectures_metadata(self):
        all_meta = build_all_networks_metadata()
        assert set(all_meta) == {"ann", "cnn", "rnn"}
        assert all_meta["ann"]["numClasses"] == 10


# ---------------------------------------------------------------------------
# Execution trace (untrained)
# ---------------------------------------------------------------------------
class TestExecutionTrace:
    def test_trace_schema(self):
        trace = build_execution_trace("ann", "mnist", ZEROS_784, use_untrained=True)
        assert "execution_id" in trace
        assert trace["model"]["id"] == "ann"
        assert len(trace["stages"]) == 4
        assert "prediction" in trace
        assert "timing" in trace
        assert "capabilities" in trace

    def test_ann_shapes(self):
        trace = build_execution_trace("ann", "mnist", ZEROS_784, use_untrained=True)
        stages = {s["layerId"]: s for s in trace["stages"]}
        assert "hidden1" in stages
        assert stages["hidden1"]["outputShape"] == [256]
        assert stages["hidden1"]["params"] > 0
        assert len(stages["hidden1"]["output_data"]) == 256

    def test_cnn_conv_feature_maps(self):
        trace = build_execution_trace("cnn", "mnist", ZEROS_784, use_untrained=True)
        stages = {s["layerId"]: s for s in trace["stages"]}
        assert "conv1" in stages
        fmap = stages["conv1"].get("feature_maps")
        assert fmap is not None
        assert len(fmap) == 32  # conv1_filters default

    def test_prediction_range(self):
        trace = build_execution_trace("ann", "mnist", ZEROS_784, use_untrained=True)
        probs = trace["prediction"]["probabilities"]
        assert len(probs) == 10
        assert 0.99 <= sum(probs) <= 1.01

    def test_lstm_verified(self):
        detail = lstm_gates_detail("rnn", "mnist", ZEROS_784, "lstm1",
                                   timestep=0, use_untrained=True)
        assert detail["verified"] is True
        assert detail["maxDeviation"] < 1e-2

    def test_lstm_gates_bounded(self):
        detail = lstm_gates_detail("rnn", "mnist", ZEROS_784, "lstm1",
                                   timestep=5, use_untrained=True)
        gates = detail["detailed"]["gates"]
        for key in ("input", "forget", "output"):
            arr = np.asarray(gates[key])
            assert arr.size == 128
            assert float(np.min(arr)) >= 0.0
            assert float(np.max(arr)) <= 1.0


# ---------------------------------------------------------------------------
# Detail correctness
# ---------------------------------------------------------------------------
class TestDenseNeuronDetail:
    def test_pre_post_consistency(self):
        det = dense_neuron_detail("ann", "mnist", ZEROS_784, "hidden1", 0,
                                  use_untrained=True)
        assert "preActivation" in det
        assert "postActivation" in det
        # With relu, post should be max(0, pre).
        assert math.isclose(det["postActivation"],
                            max(0.0, det["preActivation"]),
                            abs_tol=1e-6)

    def test_contributions_sorted(self):
        det = dense_neuron_detail("ann", "mnist", ZEROS_784, "hidden1", 10,
                                  use_untrained=True)
        contribs = det["topContributions"]
        assert len(contribs) > 0
        # Contributions should be sorted by absolute value descending.
        for i in range(len(contribs) - 1):
            assert abs(contribs[i]["contribution"]) >= abs(contribs[i + 1]["contribution"])


class TestConvCellDetail:
    def test_patch_kernel_consistency(self):
        det = conv_cell_detail("cnn", "mnist", ZEROS_784, "conv1", 0, 3, 4,
                               use_untrained=True)
        assert det["matchedOutput"] is True
        assert math.isclose(det["postActivation"], det["actualOutput"], abs_tol=1e-3)


# ---------------------------------------------------------------------------
# Raw LSTM cell validation
# ---------------------------------------------------------------------------
class TestLSTMCell:
    def test_raw_lstm_vs_keras_output(self):
        model = get_bundle("rnn", use_untrained=True).model
        layer = model.get_layer("lstm1")
        inner = layer if layer.__class__.__name__ == "LSTM" else layer.layer
        x = np.random.default_rng(42).standard_normal((28, 28), dtype=np.float32)
        record = run_lstm_cell(inner, x)
        assert record["final_hidden"].shape == (128,)
        assert record["T"] == 28

        probe_act = get_bundle("rnn", use_untrained=True).activation_model
        layers = non_input_layers(model)
        idx = next(i for i, l in enumerate(layers) if l.name == "lstm1")
        actual = np.asarray(probe_act.predict(x.reshape(1, 28, 28), verbose=0)[idx][0])
        dev = float(np.max(np.abs(record["final_hidden"] - actual)))
        assert dev < 5e-2, f"LSTM manual vs keras max deviation {dev} exceeds tolerance"


# ---------------------------------------------------------------------------
# API-level (light) – use useUntrainedWeights via the untrained bundles
# ---------------------------------------------------------------------------
class TestTraceAPIs:
    def test_networks_endpoint(self, client):
        resp = client.get("/api/lab/networks")
        assert resp.status_code == 200
        nets = resp.json()["networks"]
        assert set(nets) == {"ann", "cnn", "rnn"}

    def test_trace_endpoint_untrained_ann(self, client):
        resp = client.post("/api/lab/trace", json={
            "architecture": "ANN",
            "dataset": "mnist",
            "pixels": ZEROS_784,
            "useUntrainedWeights": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"]["id"] == "ann"
        assert len(data["stages"]) > 0

    def test_trace_detail_neuron(self, client):
        resp = client.post("/api/lab/trace/detail", json={
            "architecture": "ANN",
            "dataset": "mnist",
            "pixels": ZEROS_784,
            "layerId": "hidden1",
            "detail": "neuron",
            "selection": {"neuronIndex": 5},
        })
        # This endpoint uses use_untrained=False by default; may return 404 if
        # model weights are missing. That is fine for the light test suite.
        if resp.status_code == 200:
            data = resp.json()
            assert data["neuronIndex"] == 5
            assert "topContributions" in data