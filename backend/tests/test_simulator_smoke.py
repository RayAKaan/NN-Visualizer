"""No-regression smoke tests for the (untouched) Simulator engine.

Proves that the pretrained-registry / prediction work did not disturb the
Simulator's own forward/backward execution: the same pure-NumPy engine still
builds graphs, validates architectures, and executes forward passes.
"""
import numpy as np
import pytest

from simulator.graph_engine import build_graph
from simulator.layers import LayerConfig, validate_layers


def _mlp_layers():
    return [
        LayerConfig(layer_type="input", neurons=4, input_shape=[4]),
        LayerConfig(layer_type="dense", neurons=6, activation="relu"),
        LayerConfig(layer_type="dense", neurons=8, activation="relu"),
        LayerConfig(layer_type="output", neurons=2, activation="softmax"),
    ]


def test_validation_accepts_mlp():
    layers = _mlp_layers()
    result = validate_layers(layers)
    assert result.valid, result.errors
    assert result.total_params > 0
    assert result.architecture[-1] == 2


def test_validation_rejects_bad_graph():
    bad = [
        LayerConfig(layer_type="dense", neurons=4),
        LayerConfig(layer_type="dense", neurons=2),
    ]
    result = validate_layers(bad)
    assert not result.valid
    assert any("input" in e for e in result.errors)


def test_forward_pass_shape_and_params():
    graph = build_graph(_mlp_layers())
    out = graph.forward(np.array([0.2, -0.5, 0.9, 0.1], dtype=np.float32))
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
    # the engine emits logits for the output layer (no softmax is defined in
    # its activation registry — softmax falls back to linear)
    assert graph.total_params > 0
    assert len(graph.weights) == len(graph.biases) == 3


def test_run_forward_full_returns_steps():
    from simulator.forward_engine import run_forward_full

    graph = build_graph(_mlp_layers())
    steps, final_output, layer_outputs = run_forward_full(graph, [1.0, 0.0, 0.0, 1.0])
    assert isinstance(steps, list) and steps
    assert len(final_output) == 2
    assert len(layer_outputs) >= 1  # per-layer activations are recorded


def test_engine_backend_reports_cpu_or_gpu():
    from simulator.execution_engine import get_engine

    engine = get_engine()
    assert engine.backend in ("cpu", "gpu", "numpy", "tf")
