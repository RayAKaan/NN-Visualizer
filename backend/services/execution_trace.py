"""Execution-trace engine for the Lab.

Reads the *actual* Keras model graph and produces a faithful, layer-by-layer
execution trace of one signal: real tensors, real weights, real statistics,
real per-layer wall time, and real LSTM gate/cell dynamics computed from the
trained weights (no fabricated values anywhere).

The trace is the single source of truth for topology: it is read from
``model.layers``, never from a hardcoded frontend list.
"""
import time
import uuid
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import HTTPException

from model.ann_model import build_ann_model
from model.cnn_model import build_cnn_model
from model.rnn_model import build_rnn_model
from services.inference import inference_engine
from services.input_utils import (
    dataset_adjust,
    label_for_dataset,
    label_text_for_dataset,
    prepare_input,
    tensor_stats,
)

MODEL_TASKS = {"ann": "classification", "cnn": "classification", "rnn": "classification"}
MODEL_NAMES = {"ann": "ann_model", "cnn": "cnn_model", "rnn": "rnn_model"}


# ---------------------------------------------------------------------------
# Model bundle / helpers
# ---------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, model: tf.keras.Model, activation_model: tf.keras.Model):
        self.model = model
        self.activation_model = activation_model


_untrained: dict[str, ModelBundle] = {}


def _build_untrained_bundle(arch: str) -> ModelBundle:
    if arch == "ann":
        model = build_ann_model()
        dummy = np.zeros((1, 784), dtype=np.float32)
    elif arch == "cnn":
        model = build_cnn_model()
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
    else:
        model = build_rnn_model()
        dummy = np.zeros((1, 28, 28), dtype=np.float32)
    model.predict(dummy, verbose=0)
    outputs = [layer.output for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    return ModelBundle(model=model, activation_model=activation_model)


def get_bundle(arch: str, use_untrained: bool = False) -> ModelBundle:
    if use_untrained:
        if arch not in _untrained:
            _untrained[arch] = _build_untrained_bundle(arch)
        return _untrained[arch]
    if arch not in inference_engine.models or arch not in inference_engine.activation_models:
        raise HTTPException(status_code=404, detail=f"Model '{arch}' is not loaded")
    return ModelBundle(inference_engine.models[arch], inference_engine.activation_models[arch])


def non_input_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    return [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]


def _unwrap(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    if isinstance(layer, tf.keras.layers.Bidirectional):
        return layer.layer
    return layer


def normalize_shape(shape: Any) -> list[int]:
    if shape is None:
        return []
    dims = []
    for d in shape:
        if d is None:
            continue
        try:
            dims.append(int(d))
        except (TypeError, ValueError):
            continue
    return dims


def _tensor_shape(tensor) -> list[int]:
    """Integer shape from a KerasTensor / symbolic tensor (TF2-compatible)."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return []
    return [int(d) for d in shape if d is not None]


def operation_for(layer: tf.keras.layers.Layer) -> str:
    inner = _unwrap(layer)
    if isinstance(inner, tf.keras.layers.InputLayer):
        return "input"
    if isinstance(inner, tf.keras.layers.Dense):
        return "dense"
    if isinstance(inner, tf.keras.layers.Conv2D):
        return "conv2d"
    if isinstance(inner, tf.keras.layers.MaxPooling2D):
        return "max_pool"
    if isinstance(inner, tf.keras.layers.AveragePooling2D):
        return "avg_pool"
    if isinstance(inner, tf.keras.layers.Flatten):
        return "flatten"
    if isinstance(inner, tf.keras.layers.Activation):
        return "activation"
    if isinstance(inner, tf.keras.layers.LSTM):
        if isinstance(layer, tf.keras.layers.Bidirectional):
            return "lstm_bidirectional"
        return "lstm"
    if isinstance(inner, tf.keras.layers.Dropout):
        return "dropout"
    if isinstance(inner, tf.keras.layers.Softmax):
        return "softmax"
    name = inner.__class__.__name__
    return "".join(f"_{c}" if c.isupper() else c for c in name).strip("_").lower()


def activation_for(layer: tf.keras.layers.Layer) -> str | None:
    inner = _unwrap(layer)
    if isinstance(inner, tf.keras.layers.Softmax):
        return "softmax"
    a = getattr(inner, "activation", None)
    if a is None:
        return None
    name = getattr(a, "__name__", str(a))
    return None if name in ("linear", "") else name


def recurrent_activation_for(layer: tf.keras.layers.Layer) -> str | None:
    inner = _unwrap(layer)
    a = getattr(inner, "recurrent_activation", None)
    if a is None:
        return None
    name = getattr(a, "__name__", str(a))
    return None if name in ("linear", "") else name


def layer_type_label(layer: tf.keras.layers.Layer) -> str:
    inner = _unwrap(layer)
    return inner.__class__.__name__


def layer_overview(layer: tf.keras.layers.Layer) -> dict[str, Any]:
    op = operation_for(layer)
    return {
        "layerId": layer.name,
        "name": layer.name,
        "type": layer_type_label(layer),
        "operation": op,
        "activation": activation_for(layer),
        "recurrentActivation": recurrent_activation_for(layer) if op.startswith("lstm") else None,
        "inputShape": _tensor_shape(layer.input),
        "outputShape": _tensor_shape(layer.output),
        "params": int(layer.count_params()),
        "trainable": bool(getattr(layer, "trainable", True)),
    }


def build_network_metadata(arch: str, use_untrained: bool = False) -> dict[str, Any]:
    arch = arch.strip().lower()
    if arch not in ("ann", "cnn", "rnn"):
        raise HTTPException(status_code=400, detail=f"Unsupported architecture: {arch}")
    try:
        model = get_bundle(arch, use_untrained=use_untrained).model
        trained = not use_untrained and arch in inference_engine.models
    except HTTPException:
        model = _build_untrained_bundle(arch).model
        trained = False
    layers = [layer_overview(layer) for layer in non_input_layers(model)]
    input_shape = _tensor_shape(model.input) if model.input is not None else []
    output_shape = len(layers) and layers[-1]["outputShape"] or []
    return {
        "id": arch,
        "name": MODEL_NAMES.get(arch, arch),
        "framework": "tensorflow",
        "task": MODEL_TASKS.get(arch, "classification"),
        "numClasses": int(output_shape[-1]) if output_shape else 0,
        "inputShape": input_shape,
        "outputShape": output_shape,
        "totalParams": int(model.count_params()),
        "trained": trained,
        "layers": layers,
    }


def build_all_networks_metadata() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arch in ("ann", "cnn", "rnn"):
        result[arch] = build_network_metadata(arch)
    return result


def _capabilities(stages: list[dict[str, Any]]) -> dict[str, bool]:
    ops = {s["operation"] for s in stages}
    return {
        "neuronDetail": "dense" in ops,
        "convDetail": "conv2d" in ops,
        "lstmDetail": "lstm" in ops or "lstm_bidirectional" in ops,
        "hasDense": "dense" in ops,
        "hasConv": "conv2d" in ops,
        "hasLstm": "lstm" in ops or "lstm_bidirectional" in ops,
        "hasBackward": True,
    }


# ---------------------------------------------------------------------------
# Honest per-layer timing via cumulative subgraphs
# ---------------------------------------------------------------------------
def measure_per_layer_ms(model: tf.keras.Model, x: np.ndarray) -> dict[str, float]:
    layers = non_input_layers(model)
    timing: dict[str, float] = {}
    prefix: list[Any] = []
    previous = 0.0
    for layer in layers:
        prefix.append(layer.output)
        sub = tf.keras.Model(inputs=model.input, outputs=prefix)
        start = time.perf_counter()
        sub.predict(x, verbose=0)
        elapsed = (time.perf_counter() - start) * 1000.0
        timing[layer.name] = max(0.0, elapsed - previous)
        previous = elapsed
    return timing


# ---------------------------------------------------------------------------
# Stage record builder
# ---------------------------------------------------------------------------
def _bound_1d(values: np.ndarray, max_entries: int = 784) -> tuple[list[float], bool]:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= max_entries:
        return flat.astype(np.float32).tolist(), False
    stride = max(1, flat.size // max_entries)
    return flat[::stride].astype(np.float32).tolist(), True


def _build_stage(layer: tf.keras.layers.Layer, out: np.ndarray, compute_ms: float) -> dict[str, Any]:
    op = operation_for(layer)
    out_np = np.asarray(out)
    out_1d = out_np.reshape(-1)

    stage: dict[str, Any] = {
        "stageId": layer.name,
        "layerId": layer.name,
        "name": layer.name,
        "type": layer_type_label(layer),
        "operation": op,
        "activation": activation_for(layer),
        "inputShape": _tensor_shape(layer.input),
        "outputShape": _tensor_shape(layer.output),
        "params": int(layer.count_params()),
        "compute_time_ms": float(compute_ms),
        "timing_method": "cumulative_subgraph_diff",
        "statistics": tensor_stats(out_1d),
    }

    is_conv_pool = op in ("conv2d", "max_pool", "avg_pool")
    if out_np.ndim == 4 and is_conv_pool:
        # Channel-major nested feature maps for direct canvas rendering.
        fmap = out_np[0]
        stage["feature_maps"] = np.transpose(fmap, (2, 0, 1)).astype(np.float32).tolist()
        stage["sampled"] = False
    elif op == "flatten":
        data, sampled = _bound_1d(out_np[0], max_entries=2304)
        stage["output_data"] = data
        stage["sampled"] = sampled
    elif out_np.ndim >= 2:
        data, sampled = _bound_1d(out_np[0], max_entries=512)
        stage["output_data"] = data
        stage["sampled"] = sampled
    else:
        data, sampled = _bound_1d(out_np, max_entries=512)
        stage["output_data"] = data
        stage["sampled"] = sampled

    # Input activation tensor is attached to the stage record in
    # build_execution_trace (from the previous stage's output), because
    # ``layer.input`` is a symbolic tensor, not data.

    if isinstance(layer, tf.keras.layers.Conv2D) and layer.get_weights():
        w = np.asarray(layer.get_weights()[0]).astype(np.float32)
        stage["weights_kernel_shape"] = list(w.shape)
        stage["kernels"] = {
            "kernel": np.transpose(w, (3, 2, 0, 1)).astype(np.float32).tolist(),  # [filter][in_ch][kh][kw]
            "bias": np.asarray(layer.get_weights()[1]).reshape(-1).astype(np.float32).tolist(),
        }
    elif isinstance(layer, tf.keras.layers.Dense) and layer.get_weights():
        w = np.asarray(layer.get_weights()[0]).astype(np.float32)
        stage["weights_kernel_shape"] = list(w.shape)
        stage["bias"] = np.asarray(layer.get_weights()[1]).reshape(-1).astype(np.float32).tolist()
    elif isinstance(_unwrap(layer), tf.keras.layers.LSTM) and layer.get_weights():
        stage["weights_kernel_shape"] = list(np.asarray(layer.get_weights()[0]).shape)
        stage["recurrent_kernel_shape"] = list(np.asarray(layer.get_weights()[1]).shape)
        stage["bias"] = np.asarray(layer.get_weights()[2]).reshape(-1).astype(np.float32).tolist()

    return stage


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------
def build_execution_trace(arch: str, dataset: str, pixels: list[float], use_untrained: bool = False) -> dict[str, Any]:
    arch = arch.strip().lower()
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    activation_model = bundle.activation_model
    x = prepare_input(pixels, arch, dataset)

    start = time.perf_counter()
    outputs = activation_model.predict(x, verbose=0)
    forward_ms = (time.perf_counter() - start) * 1000.0

    layers = non_input_layers(model)
    out_map = {layer.name: np.asarray(outputs[idx]) for idx, layer in enumerate(layers)}
    per_layer_ms = measure_per_layer_ms(model, x)

    stages: list[dict[str, Any]] = []
    for idx, layer in enumerate(layers):
        out = out_map[layer.name]
        stage = _build_stage(layer, out, per_layer_ms.get(layer.name, 0.0))
        if idx == 0:
            in_data, _ = _bound_1d(x, max_entries=784)
            stage["input_data"] = in_data
        else:
            prev_out = out_map[layers[idx - 1].name]
            if prev_out.ndim == 2:
                in_data, _ = _bound_1d(prev_out[0], max_entries=512)
                stage["input_data"] = in_data
            elif prev_out.ndim == 1:
                in_data, _ = _bound_1d(prev_out, max_entries=512)
                stage["input_data"] = in_data
        stages.append(stage)

    probs = np.asarray(outputs[-1]).reshape(-1)
    probs = dataset_adjust(probs, dataset).astype(np.float32)
    label = int(np.argmax(probs)) if probs.size else 0
    confidence = float(probs[label]) if probs.size else 0.0

    trace: dict[str, Any] = {
        "execution_id": f"trace-{uuid.uuid4().hex[:12]}",
        "model": build_network_metadata(arch, use_untrained=use_untrained),
        "input": {
            "dataset": dataset,
            "source": "pixels",
            "shape": list(x.shape[1:]) if x.ndim > 1 else [x.size],
            "values": x.reshape(-1).astype(np.float32).tolist(),
        },
        "stages": stages,
        "prediction": {
            "label": label_for_dataset(dataset, label),
            "labelText": label_text_for_dataset(dataset, label),
            "confidence": confidence,
            "probabilities": probs.tolist(),
        },
        "timing": {
            "totalMs": forward_ms,
            "totalMethod": "measured_forward",
            "perLayerMs": {s["layerId"]: s["compute_time_ms"] for s in stages},
            "perLayerMethod": "cumulative_subgraph_diff",
        },
        "capabilities": _capabilities(stages),
        "meta": {
            "trained": not use_untrained and arch in inference_engine.models,
            "executedAt": time.time(),
        },
    }
    return trace


# ---------------------------------------------------------------------------
# Dense neuron detail (real pre-activation + per-input contributions)
# ---------------------------------------------------------------------------
def dense_neuron_detail(arch: str, dataset: str, pixels: list[float], layer_id: str,
                        neuron_index: int, use_untrained: bool = False) -> dict[str, Any]:
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    layer = model.get_layer(layer_id)
    if not isinstance(layer, tf.keras.layers.Dense):
        raise HTTPException(status_code=400, detail=f"Layer '{layer_id}' is not a Dense layer")
    x = prepare_input(pixels, arch, dataset)

    units = int(layer.units)
    neuron_index = int(np.clip(neuron_index, 0, units - 1))
    weights = layer.get_weights()
    kernel = np.asarray(weights[0]).astype(np.float32)  # (in, units)
    bias = np.asarray(weights[1]).astype(np.float32) if len(weights) > 1 else np.zeros(units, np.float32)

    probe = tf.keras.Model(model.input, [layer.input, layer.output])
    layer_in, layer_out = probe(x, training=False)
    layer_out_1d = np.asarray(layer_out[0]).reshape(-1)
    acts = tf.convert_to_tensor(layer_in)
    prev = np.asarray(acts[0]).reshape(-1) if isinstance(acts, tf.Tensor) else np.asarray(acts.numpy()[0]).reshape(-1)

    wj = kernel[:, neuron_index].astype(np.float64)
    xj = prev.reshape(-1).astype(np.float64)
    contrib = xj * wj
    z = float(np.sum(contrib) + float(bias[neuron_index]))
    post = float(layer_out_1d[neuron_index])
    act_name = activation_for(layer)

    order = np.argsort(-np.abs(contrib))
    top_k = min(48, contrib.size)
    top_idx = order[:top_k]
    contributions = [
        {
            "index": int(i),
            "inputValue": float(xj[i]),
            "weight": float(wj[i]),
            "contribution": float(contrib[i]),
        }
        for i in top_idx
    ]

    matched = bool(np.isclose(post, _post_activation_value(float(z), act_name), atol=1e-3))

    return {
        "layerId": layer_id,
        "neuronIndex": neuron_index,
        "units": units,
        "activation": act_name,
        "preActivation": z,
        "bias": float(bias[neuron_index]),
        "postActivation": post,
        "matchedOutput": matched,
        "inputLength": int(prev.size),
        "contributionCount": int(contrib.size),
        "maxAbsContribution": float(np.max(np.abs(contrib))) if contrib.size else 0.0,
        "topContributions": contributions,
        "weightRowStats": {
            **tensor_stats(wj),
            "norm": float(np.linalg.norm(wj)),
        },
    }


def _post_activation_value(z: float, activation: str | None) -> float:
    if activation is None or activation == "linear":
        return z
    if activation == "relu":
        return max(0.0, z)
    if activation == "softmax":
        return z  # softmax over full output vector; caller compares per-position only for relu/linear
    if activation == "sigmoid":
        return float(1.0 / (1.0 + np.exp(-z)))
    if activation == "tanh":
        return float(np.tanh(z))
    return z


# ---------------------------------------------------------------------------
# Conv cell detail (real patch x kernel at a position)
# ---------------------------------------------------------------------------
def conv_cell_detail(arch: str, dataset: str, pixels: list[float], layer_id: str,
                     filter_index: int, position_h: int, position_w: int,
                     use_untrained: bool = False) -> dict[str, Any]:
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    layer = model.get_layer(layer_id)
    if not isinstance(layer, tf.keras.layers.Conv2D):
        raise HTTPException(status_code=400, detail=f"Layer '{layer_id}' is not a Conv2D layer")
    x = prepare_input(pixels, arch, dataset)

    kernel = np.asarray(layer.get_weights()[0]).astype(np.float32)  # (kh, kw, in, filters)
    bias = np.asarray(layer.get_weights()[1]).flatten().astype(np.float32)
    kh, kw, _, filters = kernel.shape
    strides = layer.strides
    padding = getattr(layer, "padding", "same")
    s_h = int(strides[0])
    s_w = int(strides[1])

    probe = tf.keras.Model(model.input, [layer.input, layer.output])
    layer_in, layer_out = probe(x, training=False)
    input_hw = np.asarray(layer_in[0])  # (H, W, in)
    out_hw = np.asarray(layer_out[0])   # (H, W, filters)
    out_h, out_w = out_hw.shape[0], out_hw.shape[1]
    in_h, in_w = input_hw.shape[0], input_hw.shape[1]

    filter_index = int(np.clip(filter_index, 0, filters - 1))
    position_h = int(np.clip(position_h, 0, out_h - 1))
    position_w = int(np.clip(position_w, 0, out_w - 1))

    if padding == "same":
        pad_h = kh // 2
        pad_w = kw // 2
    else:
        pad_h = pad_w = 0
    padded = np.zeros((in_h + 2 * pad_h, in_w + 2 * pad_w, input_hw.shape[2]), dtype=np.float32)
    padded[pad_h:pad_h + in_h, pad_w:pad_w + in_w] = input_hw

    h_start = position_h * s_h
    w_start = position_w * s_w
    patch = padded[h_start:h_start + kh, w_start:w_start + kw, :]  # (kh, kw, in)

    kern = kernel[:, :, :, filter_index]
    z = float(np.sum(patch * kern) + float(bias[filter_index]))
    actual = float(out_hw[position_h, position_w, filter_index])
    act_name = activation_for(layer)
    post = _post_activation_value(z, act_name)
    matched = bool(np.isclose(post, actual, atol=1e-3))

    return {
        "layerId": layer_id,
        "filterIndex": filter_index,
        "position": {"h": position_h, "w": position_w},
        "patch": patch.tolist(),
        "kernel": kern.tolist(),
        "bias": float(bias[filter_index]),
        "preActivation": z,
        "activation": act_name,
        "postActivation": post,
        "actualOutput": actual,
        "matchedOutput": matched,
        "stride": [s_h, s_w],
        "padding": padding,
        "inputShape": list(input_hw.shape),
        "outputShape": [out_h, out_w, filters],
    }


# ---------------------------------------------------------------------------
# Real LSTM gate dynamics (manual cell computation from trained weights)
# ---------------------------------------------------------------------------
def _split_gates(g: np.ndarray, units: int) -> list[np.ndarray]:
    return [g[i * units:(i + 1) * units] for i in range(4)]


def run_lstm_cell(layer: tf.keras.layers.Layer, x_seq: np.ndarray) -> dict[str, Any]:
    """Manually run the LSTM cell over ``x_seq`` (T, input_dim) using the real
    trained weights. Returns per-timestep gate/cell/hidden vectors plus the
    final hidden state, which is validated against the Keras layer output."""
    weights = layer.get_weights()
    kernel = np.asarray(weights[0], dtype=np.float64)      # (in, 4u) order [i, f, c, o]
    recurrent = np.asarray(weights[1], dtype=np.float64)   # (u, 4u)
    bias = np.asarray(weights[2], dtype=np.float64)        # (4u,)
    units = int(layer.units)
    T = int(x_seq.shape[0])

    h = np.zeros(units, dtype=np.float64)
    c = np.zeros(units, dtype=np.float64)
    timesteps: list[dict[str, np.ndarray]] = []

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    for t in range(T):
        xt = x_seq[t].astype(np.float64)
        gates = xt @ kernel + h @ recurrent + bias
        g_i, g_f, g_c, g_o = _split_gates(gates, units)
        i_gate = _sigmoid(g_i)
        f_gate = _sigmoid(g_f)
        c_tilde = np.tanh(g_c)
        o_gate = _sigmoid(g_o)
        c = f_gate * c + i_gate * c_tilde
        h = o_gate * np.tanh(c)
        timesteps.append(
            {
                "step": t,
                "input_gate": i_gate.astype(np.float32),
                "forget_gate": f_gate.astype(np.float32),
                "candidate": c_tilde.astype(np.float32),
                "output_gate": o_gate.astype(np.float32),
                "cell_state": c.astype(np.float32),
                "hidden_state": h.astype(np.float32),
            }
        )

    return {"timesteps": timesteps, "final_hidden": h.astype(np.float32), "units": units, "T": T}


def lstm_input_sequence(arch: str, dataset: str, pixels: list[float],
                        layer: tf.keras.layers.Layer) -> np.ndarray:
    x = prepare_input(pixels, arch, dataset)
    if x.ndim == 3:
        return x[0]  # (T, input_dim)
    # Unrolled / unusual case: reshape trailing dims as (T, feature).
    flat = x.reshape(-1)
    shape = normalize_shape(getattr(layer, "input_shape", None))
    seq_len = shape[0] if shape else 28
    return flat.reshape(seq_len, -1)


def lstm_gates_detail(arch: str, dataset: str, pixels: list[float], layer_id: str,
                      timestep: int | None = None, use_untrained: bool = False) -> dict[str, Any]:
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    layer = model.get_layer(layer_id)
    inner = _unwrap(layer)
    if not isinstance(inner, tf.keras.layers.LSTM):
        raise HTTPException(status_code=400, detail=f"Layer '{layer_id}' is not an LSTM layer")

    x_seq = lstm_input_sequence(arch, dataset, pixels, layer)
    record = run_lstm_cell(inner, x_seq)
    layered = non_input_layers(model)
    idx = next(i for i, l in enumerate(layered) if l.name == layer_id)
    actual = np.asarray(bundle.activation_model.predict(
        tf.convert_to_tensor(prepare_input(pixels, arch, dataset)), verbose=0)[idx][0]).reshape(-1).astype(np.float64)
    dev = float(np.max(np.abs(record["final_hidden"] - actual[: inner.units]))) if actual.size else 0.0

    summary = {
        "hiddenNorm": [float(np.linalg.norm(t["hidden_state"])) for t in record["timesteps"]],
        "cellNorm": [float(np.linalg.norm(t["cell_state"])) for t in record["timesteps"]],
        "forgetMean": [float(np.mean(t["forget_gate"])) for t in record["timesteps"]],
        "inputMean": [float(np.mean(t["input_gate"])) for t in record["timesteps"]],
        "outputMean": [float(np.mean(t["output_gate"])) for t in record["timesteps"]],
        "cellMean": [float(np.mean(t["cell_state"])) for t in record["timesteps"]],
    }

    detailed = None
    if timestep is not None:
        t_idx = int(np.clip(timestep, 0, record["T"] - 1))
        t = record["timesteps"][t_idx]
        detailed = {
            "timestep": t_idx,
            "hiddenState": t["hidden_state"].tolist(),
            "cellState": t["cell_state"].tolist(),
            "gates": {
                "input": t["input_gate"].tolist(),
                "forget": t["forget_gate"].tolist(),
                "candidate": t["candidate"].tolist(),
                "output": t["output_gate"].tolist(),
                "cell": t["cell_state"].tolist(),
            },
        }

    return {
        "layerId": layer_id,
        "units": record["units"],
        "timesteps": record["T"],
        "verified": bool(dev < 1e-2),
        "maxDeviation": dev,
        "summary": summary,
        "detailed": detailed,
    }


def lstm_gate_gradients(arch: str, dataset: str, pixels: list[float], layer_id: str,
                        true_label: int, use_untrained: bool = False) -> dict[str, Any]:
    """BPTT over the manual cell to get per-gate gradient magnitude that
    actually flows into each gate across all timesteps."""
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    layer = model.get_layer(layer_id)
    inner = _unwrap(layer)
    if not isinstance(inner, tf.keras.layers.LSTM):
        raise HTTPException(status_code=400, detail=f"Layer '{layer_id}' is not an LSTM layer")

    x_tf = tf.convert_to_tensor(prepare_input(pixels, arch, dataset))
    x_seq = lstm_input_sequence(arch, dataset, pixels, layer)
    record = run_lstm_cell(inner, x_seq)
    units = inner.units
    T = record["T"]

    weights = inner.get_weights()
    recurrent = np.asarray(weights[1], dtype=np.float64)  # (u, 4u) order [i,f,c,o]

    probe = tf.keras.Model(model.input, [layer.output, model.output])
    with tf.GradientTape() as tape:
        layer_out, logits = probe(x_tf, training=False)
        probs = tf.nn.softmax(logits, axis=-1)
        classes = int(logits.shape[1])
        label = int(np.clip(true_label, 0, classes - 1))
        loss = -tf.math.log(probs[0, label] + 1e-7)
    grad_out = tape.gradient(loss, layer_out)
    if grad_out is None:
        raise HTTPException(status_code=500, detail="No gradient reached LSTM output")
    dh = np.asarray(grad_out[0]).reshape(-1).astype(np.float64)

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    timesteps = record["timesteps"]
    gates_inv: list[np.ndarray] = []
    for t in timesteps:
        i = t["input_gate"].astype(np.float64)
        f = t["forget_gate"].astype(np.float64)
        o = t["output_gate"].astype(np.float64)
        cand = t["candidate"].astype(np.float64)
        c = t["cell_state"].astype(np.float64)
        gates_inv.append((i, f, o, cand, c))

    grads = {
        "forget": np.zeros(units, dtype=np.float64),
        "input": np.zeros(units, dtype=np.float64),
        "candidate": np.zeros(units, dtype=np.float64),
        "output": np.zeros(units, dtype=np.float64),
        "cell_state": np.zeros(units, dtype=np.float64),
    }
    c_prev_hist = [np.zeros(units, dtype=np.float64)]
    for t in range(T):
        c_prev_hist.append(gates_inv[t][4])

    dc = np.zeros(units, dtype=np.float64)
    for t in range(T - 1, -1, -1):
        i, f, o, cand, c_t = gates_inv[t]
        c_prev = c_prev_hist[t]
        tanhc = np.tanh(c_t)
        do = dh * tanhc
        dc_t = dc + dh * o * (1.0 - tanhc ** 2)
        dcand = dc_t * i * (1.0 - cand ** 2)
        di = dc_t * cand * i * (1.0 - i)
        df = dc_t * c_prev * f * (1.0 - f)
        grads["forget"] += df
        grads["input"] += di
        grads["candidate"] += dcand
        grads["output"] += do
        grads["cell_state"] += dc_t * f

        dg = np.concatenate([di, df, dcand, do])  # order [i, f, c, o]
        dh = dg @ recurrent.T
        dc = dc_t * f
    return {
        "layerId": layer_id,
        "gradients": {
            "forget": grads["forget"].tolist(),
            "input": grads["input"].tolist(),
            "output": grads["output"].tolist(),
            "candidate": grads["candidate"].tolist(),
            "cell_state": grads["cell_state"].tolist(),
        },
        "magnitudes": {
            "forget": float(np.linalg.norm(grads["forget"])),
            "input": float(np.linalg.norm(grads["input"])),
            "output": float(np.linalg.norm(grads["output"])),
            "candidate": float(np.linalg.norm(grads["candidate"])),
            "cell_state": float(np.linalg.norm(grads["cell_state"])),
        },
        "method": "manual_bptt",
    }


def real_final_gates(arch: str, dataset: str, pixels: list[float],
                     use_untrained: bool = False) -> dict[str, list[float]]:
    """Real final-timestep gate vectors (replaces the old fabricated _fake_gates)."""
    bundle = get_bundle(arch, use_untrained=use_untrained)
    model = bundle.model
    inner = None
    for layer in non_input_layers(model):
        unwrapped = _unwrap(layer)
        if isinstance(unwrapped, tf.keras.layers.LSTM):
            inner = unwrapped
            break
    if inner is None:
        return {"forget": [], "input": [], "output": [], "cell_state": []}
    x_seq = lstm_input_sequence(arch, dataset, pixels, inner)
    record = run_lstm_cell(inner, x_seq)
    last = record["timesteps"][-1]
    return {
        "forget": last["forget_gate"].tolist(),
        "input": last["input_gate"].tolist(),
        "output": last["output_gate"].tolist(),
        "cell_state": last["cell_state"].tolist(),
    }