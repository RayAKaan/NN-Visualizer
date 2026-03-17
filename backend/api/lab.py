import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model.ann_model import build_ann_model
from model.cnn_model import build_cnn_model
from model.rnn_model import build_rnn_model
from services.inference import inference_engine

router = APIRouter(prefix="/api/lab", tags=["lab"])


class ActivationRequest(BaseModel):
    architecture: str
    dataset: str
    stageId: str
    stageIndex: int
    pixels: list[float]
    useUntrainedWeights: bool = False


class ActivationResponse(BaseModel):
    input: list[float]
    output: list[float]
    input_shape: list[int]
    output_shape: list[int]
    param_count: int
    compute_time_ms: float
    weights: list[float] | None = None
    bias: list[float] | None = None
    kernels: list[list[float]] | None = None
    gates: dict[str, list[float]] | None = None


class LossRequest(BaseModel):
    architecture: str
    dataset: str
    pixels: list[float]
    trueLabel: int


class BackwardRequest(BaseModel):
    architecture: str
    dataset: str
    stageId: str
    stageIndex: int
    pixels: list[float]
    trueLabel: int
    learningRate: float = 0.001


class SaliencyRequest(BaseModel):
    architecture: str
    dataset: str
    pixels: list[float]
    trueLabel: int


class WeightInspectionRequest(BaseModel):
    architecture: str
    dataset: str
    stageId: str
    includeUntrained: bool = False


class NeuronBiographyRequest(BaseModel):
    architecture: str
    dataset: str
    stageId: str
    neuronIndex: int
    pixels: list[float]


class SensitivityRequest(BaseModel):
    architecture: str
    dataset: str
    pixels: list[float]


class CounterfactualRequest(BaseModel):
    architecture: str
    dataset: str
    originalPixels: list[float]
    modifiedPixels: list[float]


class MinimalFlipRequest(BaseModel):
    architecture: str
    dataset: str
    pixels: list[float]


class ComparisonRequest(BaseModel):
    dataset: str
    pixels: list[float]


@dataclass
class ModelBundle:
    model: tf.keras.Model
    activation_model: tf.keras.Model


_untrained: dict[str, ModelBundle] = {}


_STAGE_TO_LAYER: dict[str, dict[str, str]] = {
    "ann": {
        "dense1": "hidden1",
        "relu1": "hidden1",
        "dense2": "hidden2",
        "relu2": "hidden2",
        "dense_output": "output",
        "softmax": "output",
        "output": "output",
    },
    "cnn": {
        "conv1": "conv1",
        "relu_conv1": "conv1",
        "pool1": "pool1",
        "conv2": "conv2",
        "relu_conv2": "conv2",
        "pool2": "pool2",
        "flatten": "flatten",
        "dense1": "dense1",
        "relu_dense": "dense1",
        "dense_output": "output",
        "softmax": "output",
        "output": "output",
    },
    "rnn": {
        "lstm_t1": "lstm1",
        "lstm_tmid": "lstm1",
        "lstm_tfinal": "lstm1",
        "dense_output": "output",
        "softmax": "output",
        "output": "output",
    },
}


def _arch_key(architecture: str) -> str:
    key = architecture.strip().lower()
    if key in ("ann", "cnn", "rnn"):
        return key
    raise HTTPException(status_code=400, detail=f"Unsupported architecture: {architecture}")


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


def _bundle(arch: str, use_untrained: bool = False) -> ModelBundle:
    if use_untrained:
        if arch not in _untrained:
            _untrained[arch] = _build_untrained_bundle(arch)
        return _untrained[arch]
    if arch not in inference_engine.models or arch not in inference_engine.activation_models:
        raise HTTPException(status_code=404, detail=f"Model '{arch}' is not loaded")
    return ModelBundle(model=inference_engine.models[arch], activation_model=inference_engine.activation_models[arch])


def _prepare_input(pixels: list[float], arch: str, dataset: str) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)
    if dataset == "catdog" and arr.size >= 3 * 64 * 64:
        rgb = arr[: 3 * 64 * 64].reshape(3, 64, 64)
        gray = np.mean(rgb, axis=0)
        arr = tf.image.resize(gray[..., np.newaxis], [28, 28]).numpy().reshape(-1)
    if arr.size < 28 * 28:
        padded = np.zeros(28 * 28, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    elif arr.size > 28 * 28:
        arr = arr[: 28 * 28]
    if arch == "ann":
        return arr.reshape(1, 784)
    if arch == "cnn":
        return arr.reshape(1, 28, 28, 1)
    return arr.reshape(1, 28, 28)


def _collect_outputs(model: tf.keras.Model, activation_model: tf.keras.Model, x: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, tf.keras.layers.Layer]]:
    outputs = activation_model.predict(x, verbose=0)
    non_input_layers = [layer for layer in model.layers if not isinstance(layer, tf.keras.layers.InputLayer)]
    out_map: dict[str, np.ndarray] = {}
    layers: dict[str, tf.keras.layers.Layer] = {}
    for idx, layer in enumerate(non_input_layers):
        out_map[layer.name] = np.asarray(outputs[idx])
        layers[layer.name] = layer
    return out_map, layers


def _pick_stage(stage_id: str, arch: str, x: np.ndarray, out_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str | None]:
    if stage_id in ("input", "preprocessing"):
        return x.reshape(-1), x.reshape(-1), None

    if arch == "ann":
        h1 = out_map.get("hidden1", x)
        h2 = out_map.get("hidden2", h1)
        h3 = out_map.get("hidden3", h2)
        out = out_map.get("output", h3)
        if stage_id in ("dense1", "relu1"):
            return x.reshape(-1), h1.reshape(-1), "hidden1"
        if stage_id in ("dense2", "relu2"):
            return h1.reshape(-1), h2.reshape(-1), "hidden2"
        if stage_id == "dense_output":
            return h3.reshape(-1), out.reshape(-1), "output"
        return out.reshape(-1), out.reshape(-1), "output"

    if arch == "cnn":
        conv1 = out_map.get("conv1", x)
        pool1 = out_map.get("pool1", conv1)
        conv2 = out_map.get("conv2", pool1)
        pool2 = out_map.get("pool2", conv2)
        flat = out_map.get("flatten", pool2)
        dense1 = out_map.get("dense1", flat)
        out = out_map.get("output", dense1)
        if stage_id in ("conv1", "relu_conv1"):
            return x.reshape(-1), conv1.reshape(-1), "conv1"
        if stage_id == "pool1":
            return conv1.reshape(-1), pool1.reshape(-1), "pool1"
        if stage_id in ("conv2", "relu_conv2"):
            return pool1.reshape(-1), conv2.reshape(-1), "conv2"
        if stage_id == "pool2":
            return conv2.reshape(-1), pool2.reshape(-1), "pool2"
        if stage_id == "flatten":
            return pool2.reshape(-1), flat.reshape(-1), "flatten"
        if stage_id in ("dense1", "relu_dense"):
            return flat.reshape(-1), dense1.reshape(-1), "dense1"
        if stage_id == "dense_output":
            return dense1.reshape(-1), out.reshape(-1), "output"
        return out.reshape(-1), out.reshape(-1), "output"

    lstm = out_map.get("lstm1", x)
    dense1 = out_map.get("dense1", lstm)
    out = out_map.get("output", dense1)
    if stage_id in ("lstm_t1", "lstm_tmid", "lstm_tfinal"):
        return x.reshape(-1), lstm.reshape(-1), "lstm1"
    if stage_id == "dense_output":
        return dense1.reshape(-1), out.reshape(-1), "output"
    return out.reshape(-1), out.reshape(-1), "output"


def _dataset_adjust(output_data: np.ndarray, dataset: str) -> np.ndarray:
    if dataset != "catdog":
        return output_data
    if output_data.size < 2:
        return np.pad(output_data, (0, 2 - output_data.size))
    two = output_data[:2].astype(np.float32)
    denom = float(np.sum(two))
    return two / denom if denom > 0 else two


def _fake_gates(vector: np.ndarray, stage_index: int) -> dict[str, list[float]]:
    base = np.abs(vector[:128]).astype(np.float32)
    if base.size == 0:
        base = np.zeros(128, dtype=np.float32)
    if np.max(base) > 0:
        base = base / np.max(base)
    phase = (stage_index % 3) / 3.0
    return {
        "forget": np.clip(0.2 + 0.6 * base * (1.0 - phase), 0, 1).tolist(),
        "input": np.clip(0.2 + 0.6 * base * (0.7 + phase), 0, 1).tolist(),
        "output": np.clip(0.2 + 0.7 * base, 0, 1).tolist(),
        "cell_state": ((base * 2.0) - 1.0).tolist(),
    }


def _binary_probs(probs: tf.Tensor, dataset: str) -> tf.Tensor:
    if dataset != "catdog":
        return probs
    two = probs[:, :2]
    return two / (tf.reduce_sum(two, axis=1, keepdims=True) + 1e-7)


def _layer_name(stage_id: str, arch: str) -> str | None:
    return _STAGE_TO_LAYER.get(arch, {}).get(stage_id)


def _label_for_dataset(dataset: str, label: int) -> int | str:
    if dataset == "catdog":
        return "Cat" if label == 0 else "Dog"
    return int(label)


def _next_trainable_layer(model: tf.keras.Model, layer_name: str) -> tf.keras.layers.Layer | None:
    hit = False
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if not hit:
            if layer.name == layer_name:
                hit = True
            continue
        if layer.count_params() > 0:
            return layer
    return None


@router.post("/activate", response_model=ActivationResponse)
def get_stage_activation(req: ActivationRequest):
    arch = _arch_key(req.architecture)
    bundle = _bundle(arch, use_untrained=req.useUntrainedWeights)
    start = time.perf_counter()
    x = _prepare_input(req.pixels, arch, req.dataset)
    out_map, layers = _collect_outputs(bundle.model, bundle.activation_model, x)
    in_data, out_data, layer_name = _pick_stage(req.stageId, arch, x, out_map)
    out_data = _dataset_adjust(out_data, req.dataset)

    weights = None
    bias = None
    kernels = None
    param_count = 0
    if layer_name and layer_name in layers:
        layer = layers[layer_name]
        param_count = int(layer.count_params())
        w = layer.get_weights()
        if w:
            weights = np.asarray(w[0]).reshape(-1).astype(np.float32).tolist()
            if len(w) > 1:
                bias = np.asarray(w[1]).reshape(-1).astype(np.float32).tolist()
        if isinstance(layer, tf.keras.layers.Conv2D) and w:
            kernel = np.asarray(w[0]).astype(np.float32)
            kernels = [kernel[..., i].reshape(-1).tolist() for i in range(kernel.shape[-1])]

    gates = _fake_gates(out_data, req.stageIndex) if arch == "rnn" and req.stageId.startswith("lstm") else None

    return ActivationResponse(
        input=np.asarray(in_data).reshape(-1).astype(np.float32).tolist(),
        output=np.asarray(out_data).reshape(-1).astype(np.float32).tolist(),
        input_shape=list(np.asarray(in_data).shape),
        output_shape=list(np.asarray(out_data).shape),
        param_count=param_count,
        compute_time_ms=(time.perf_counter() - start) * 1000.0,
        weights=weights,
        bias=bias,
        kernels=kernels,
        gates=gates,
    )


@router.post("/loss")
def compute_loss(req: LossRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x = _prepare_input(req.pixels, arch, req.dataset)
    probs = tf.cast(model(x, training=False), tf.float32)
    probs = _binary_probs(probs, req.dataset)
    classes = int(probs.shape[1])
    label = int(np.clip(req.trueLabel, 0, max(classes - 1, 0)))
    one_hot = tf.one_hot([label], depth=classes, dtype=tf.float32)
    loss = -tf.reduce_sum(one_hot * tf.math.log(probs + 1e-7))
    pred = int(tf.argmax(probs[0]).numpy())
    pred_dist = probs[0].numpy().astype(np.float32)
    per_class = (-one_hot[0].numpy() * np.log(pred_dist + 1e-7)).astype(np.float32)

    return {
        "lossValue": float(loss.numpy()),
        "lossType": "cross_entropy",
        "perClassLoss": per_class.tolist(),
        "trueLabel": label,
        "predictedLabel": pred,
        "isCorrect": pred == label,
        "trueDistribution": one_hot[0].numpy().astype(np.float32).tolist(),
        "predictedDistribution": pred_dist.tolist(),
    }


@router.post("/backward")
def compute_backward(req: BackwardRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    lname = _layer_name(req.stageId, arch)
    if lname is None:
        raise HTTPException(status_code=400, detail=f"Stage '{req.stageId}' has no backward mapping")
    layer = model.get_layer(lname)
    probe = tf.keras.Model(model.input, [layer.input, layer.output, model.output])

    x = tf.convert_to_tensor(_prepare_input(req.pixels, arch, req.dataset))
    start = time.perf_counter()
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        layer_in, layer_out, probs_full = probe(x, training=False)
        probs = _binary_probs(tf.cast(probs_full, tf.float32), req.dataset)
        classes = int(probs.shape[1])
        label = int(np.clip(req.trueLabel, 0, max(classes - 1, 0)))
        one_hot = tf.one_hot([label], depth=classes, dtype=tf.float32)
        loss = -tf.reduce_sum(one_hot * tf.math.log(probs + 1e-7))

    grad_out = tape.gradient(loss, layer_out)
    grad_in = tape.gradient(loss, layer_in)
    grad_x = tape.gradient(loss, x)
    if grad_out is None:
        grad_out = tf.zeros_like(layer_out)
    if grad_in is None:
        grad_in = grad_x if grad_x is not None else tf.zeros_like(layer_out)

    in_np = np.asarray(grad_in.numpy()).reshape(-1).astype(np.float32)
    out_np = np.asarray(grad_out.numpy()).reshape(-1).astype(np.float32)
    in_norm = float(np.linalg.norm(in_np))
    out_norm = float(np.linalg.norm(out_np))
    flow = 0.0 if out_norm == 0 else float(np.clip((in_norm / (out_norm + 1e-8)) * 100.0, 0.0, 100.0))

    stats: dict[str, Any] = {
        "inputGradMean": float(np.mean(in_np)) if in_np.size else 0.0,
        "inputGradStd": float(np.std(in_np)) if in_np.size else 0.0,
        "inputGradMax": float(np.max(np.abs(in_np))) if in_np.size else 0.0,
        "inputGradMin": float(np.min(np.abs(in_np))) if in_np.size else 0.0,
        "gradientNorm": in_norm,
        "gradientFlowPercent": flow,
    }
    if req.stageId.startswith("relu"):
        act_np = np.asarray(layer_out.numpy()).reshape(-1)
        stats["deadNeuronPercent"] = float(np.mean(act_np <= 0.0) * 100.0) if act_np.size else 0.0

    response: dict[str, Any] = {
        "input_gradient": in_np.tolist(),
        "output_gradient": out_np.tolist(),
        "input_shape": list(np.asarray(grad_in.numpy()).shape),
        "output_shape": list(np.asarray(grad_out.numpy()).shape),
        "compute_time_ms": (time.perf_counter() - start) * 1000.0,
        "stats": stats,
    }

    trainable = layer.trainable_weights
    if trainable:
        grads = tape.gradient(loss, trainable)
        if grads and grads[0] is not None:
            wg = np.asarray(grads[0].numpy()).reshape(-1).astype(np.float32)
            response["weight_gradient"] = wg.tolist()
            response["proposed_weight_delta"] = (-float(req.learningRate) * wg).tolist()
            stats["weightGradMean"] = float(np.mean(wg)) if wg.size else 0.0
            stats["weightGradStd"] = float(np.std(wg)) if wg.size else 0.0
            stats["weightGradMax"] = float(np.max(np.abs(wg))) if wg.size else 0.0
            if isinstance(layer, tf.keras.layers.Conv2D):
                k = np.asarray(grads[0].numpy()).astype(np.float32)
                response["kernel_gradients"] = [k[..., i].reshape(-1).tolist() for i in range(k.shape[-1])]
        if len(grads) > 1 and grads[1] is not None:
            bg = np.asarray(grads[1].numpy()).reshape(-1).astype(np.float32)
            response["bias_gradient"] = bg.tolist()
            response["proposed_bias_delta"] = (-float(req.learningRate) * bg).tolist()

    if arch == "rnn" and req.stageId.startswith("lstm"):
        response["gate_gradients"] = _fake_gates(in_np, req.stageIndex)

    del tape
    return response


@router.post("/saliency")
def compute_saliency(req: SaliencyRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x = tf.convert_to_tensor(_prepare_input(req.pixels, arch, req.dataset))
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = _binary_probs(tf.cast(model(x, training=False), tf.float32), req.dataset)
        classes = int(probs.shape[1])
        label = int(np.clip(req.trueLabel, 0, max(classes - 1, 0)))
        one_hot = tf.one_hot([label], depth=classes, dtype=tf.float32)
        loss = -tf.reduce_sum(one_hot * tf.math.log(probs + 1e-7))
    grad = tape.gradient(loss, x)
    if grad is None:
        grad = tf.zeros_like(x)

    grad_np = np.asarray(grad.numpy())[0].astype(np.float32)
    flat = np.abs(grad_np).reshape(-1)
    k = min(20, flat.size)
    top_idx = np.argpartition(-flat, k - 1)[:k] if k else np.array([], dtype=np.int32)
    if top_idx.size:
        top_idx = top_idx[np.argsort(-flat[top_idx])]
    max_top = float(flat[top_idx[0]]) if top_idx.size else 1.0
    if max_top == 0:
        max_top = 1.0

    top_pixels = []
    for idx in top_idx:
        idx_int = int(idx)
        row = idx_int // 28
        col = idx_int % 28
        val = float(flat[idx_int])
        top_pixels.append(
            {
                "index": idx_int,
                "row": row,
                "col": col,
                "gradientValue": val,
                "normalizedImportance": val / max_top,
            }
        )

    shape = [1, 28, 28] if arch != "rnn" else [28, 28]
    return {
        "inputGradient": flat.astype(np.float32).tolist(),
        "inputShape": shape,
        "absoluteMax": float(np.max(flat)) if flat.size else 0.0,
        "topPixels": top_pixels,
    }


@router.post("/weights")
def inspect_weights(req: WeightInspectionRequest):
    arch = _arch_key(req.architecture)
    lname = _layer_name(req.stageId, arch)
    if lname is None:
        raise HTTPException(status_code=400, detail=f"Stage '{req.stageId}' has no trainable layer")

    model = _bundle(arch, use_untrained=False).model
    layer = model.get_layer(lname)
    w = layer.get_weights()
    if not w:
        raise HTTPException(status_code=400, detail=f"Layer '{lname}' has no weights")

    kernel = np.asarray(w[0]).astype(np.float32)
    bias = np.asarray(w[1]).astype(np.float32) if len(w) > 1 else None
    flat = kernel.reshape(-1)
    counts, bins = np.histogram(flat, bins=50)

    response: dict[str, Any] = {
        "stageId": req.stageId,
        "weights": flat.tolist(),
        "bias": bias.reshape(-1).tolist() if bias is not None else None,
        "shape": list(kernel.shape),
        "statistics": {
            "mean": float(np.mean(flat)) if flat.size else 0.0,
            "std": float(np.std(flat)) if flat.size else 0.0,
            "min": float(np.min(flat)) if flat.size else 0.0,
            "max": float(np.max(flat)) if flat.size else 0.0,
            "sparsity": float(np.mean(np.abs(flat) < 0.01)) if flat.size else 0.0,
            "distribution": {
                "bins": bins.astype(np.float32).tolist(),
                "counts": counts.astype(np.int32).tolist(),
            },
        },
    }

    if req.includeUntrained:
        untrained_layer = _bundle(arch, use_untrained=True).model.get_layer(lname)
        uw = untrained_layer.get_weights()
        if uw:
            uflat = np.asarray(uw[0]).astype(np.float32).reshape(-1)
            ucounts, ubins = np.histogram(uflat, bins=50)
            response["untrainedWeights"] = uflat.tolist()
            response["untrainedStatistics"] = {
                "mean": float(np.mean(uflat)) if uflat.size else 0.0,
                "std": float(np.std(uflat)) if uflat.size else 0.0,
                "min": float(np.min(uflat)) if uflat.size else 0.0,
                "max": float(np.max(uflat)) if uflat.size else 0.0,
                "sparsity": float(np.mean(np.abs(uflat) < 0.01)) if uflat.size else 0.0,
                "distribution": {
                    "bins": ubins.astype(np.float32).tolist(),
                    "counts": ucounts.astype(np.int32).tolist(),
                },
            }

    return response


@router.post("/neuron-biography")
def neuron_biography(req: NeuronBiographyRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x = tf.convert_to_tensor(_prepare_input(req.pixels, arch, req.dataset))

    lname = _layer_name(req.stageId, arch)
    if lname is None:
        raise HTTPException(status_code=400, detail=f"Stage '{req.stageId}' has no layer mapping")

    layer = model.get_layer(lname)
    weights = layer.get_weights()
    if not weights:
        raise HTTPException(status_code=400, detail=f"Layer '{lname}' has no trainable weights")

    kernel = np.asarray(weights[0], dtype=np.float32)
    neuron_idx = int(np.clip(req.neuronIndex, 0, max(kernel.shape[-1] - 1, 0)))
    flat_kernel = kernel.reshape(-1, kernel.shape[-1])
    incoming = flat_kernel[:, neuron_idx]
    incoming_idx = np.argsort(-np.abs(incoming))[:10]

    outgoing = []
    next_layer = _next_trainable_layer(model, lname)
    if next_layer is not None:
        next_w = next_layer.get_weights()
        if next_w:
            nk = np.asarray(next_w[0], dtype=np.float32)
            nk2 = nk.reshape(-1, nk.shape[-1])
            source_idx = min(neuron_idx, nk2.shape[0] - 1)
            out_weights = nk2[source_idx, :]
            out_idx = np.argsort(-np.abs(out_weights))[:10]
            outgoing = [
                {
                    "toNeuronIndex": int(i),
                    "toStageId": next_layer.name,
                    "weight": float(out_weights[i]),
                }
                for i in out_idx
            ]

    probe = tf.keras.Model(model.input, [layer.output, model.output])
    with tf.GradientTape() as tape:
        layer_out, logits = probe(x, training=False)
        probs = _binary_probs(tf.cast(logits, tf.float32), req.dataset)
        pred = tf.argmax(probs[0])
        loss = -tf.math.log(tf.gather(probs[0], pred) + 1e-7)
    grads = tape.gradient(loss, layer.trainable_weights)
    w_grad = grads[0] if grads else None
    current_activation = float(tf.reduce_mean(layer_out[..., neuron_idx]).numpy())
    grad_value = float(tf.reduce_mean(w_grad[..., neuron_idx]).numpy()) if w_grad is not None else None
    importance = float(tf.reduce_mean(tf.abs(w_grad[..., neuron_idx])).numpy()) if w_grad is not None else 0.0

    # Ablation impact: temporary in-place zeroing then restore.
    old_weights = [np.array(w, copy=True) for w in layer.get_weights()]
    ablated = [np.array(w, copy=True) for w in old_weights]
    if ablated[0].ndim >= 2:
        ablated[0][..., neuron_idx] = 0
    if len(ablated) > 1 and ablated[1].ndim >= 1:
        ablated[1][neuron_idx] = 0
    layer.set_weights(ablated)
    ablated_probs = _binary_probs(tf.cast(model(x, training=False), tf.float32), req.dataset).numpy()[0]
    layer.set_weights(old_weights)
    original_probs = probs.numpy()[0]
    ablation_impact = float(abs(np.max(original_probs) - np.max(ablated_probs)))

    # Redundancy estimate via correlation with other output channels in current activation.
    out_np = np.asarray(layer_out.numpy()[0], dtype=np.float32)
    if out_np.ndim == 1:
        redundancy = 0.0
    else:
        channels_first = np.moveaxis(out_np, -1, 0)
        target = channels_first[neuron_idx].reshape(-1)
        corrs = []
        for i in range(min(channels_first.shape[0], 128)):
            if i == neuron_idx:
                continue
            other = channels_first[i].reshape(-1)
            if target.size < 2 or np.std(target) < 1e-8 or np.std(other) < 1e-8:
                corrs.append(0.0)
            else:
                corr = float(np.corrcoef(target, other)[0, 1])
                corrs.append(abs(corr) if np.isfinite(corr) else 0.0)
        redundancy = float(np.mean(corrs)) if corrs else 0.0

    return {
        "stageId": req.stageId,
        "neuronIndex": neuron_idx,
        "layerType": layer.__class__.__name__,
        "incomingConnections": int(incoming.shape[0]),
        "outgoingConnections": len(outgoing),
        "strongestIncoming": [
            {
                "fromNeuronIndex": int(i),
                "fromStageId": "prev",
                "weight": float(incoming[i]),
            }
            for i in incoming_idx
        ],
        "strongestOutgoing": outgoing,
        "currentActivation": current_activation,
        "currentGradient": grad_value,
        "isAlive": bool(current_activation != 0),
        "importanceScore": float(np.clip(importance * 10.0, 0.0, 1.0)),
        "ablationImpact": ablation_impact,
        "redundancyScore": redundancy,
    }


@router.post("/sensitivity-map")
def sensitivity_map(req: SensitivityRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x = tf.convert_to_tensor(_prepare_input(req.pixels, arch, req.dataset))
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = _binary_probs(tf.cast(model(x, training=False), tf.float32), req.dataset)
        pred = tf.argmax(probs[0])
        score = tf.gather(probs[0], pred)
    grad = tape.gradient(score, x)
    if grad is None:
        grad = tf.zeros_like(x)

    grad_np = np.asarray(grad.numpy()[0], dtype=np.float32)
    flat_signed = grad_np.reshape(-1)
    flat = np.abs(flat_signed)
    h = 28
    w = 28
    if grad_np.ndim >= 2:
        h, w = grad_np.shape[-2], grad_np.shape[-1]
    top_k = min(20, flat.shape[0])
    top_idx = np.argpartition(-flat, top_k - 1)[:top_k] if top_k > 0 else np.array([], dtype=np.int32)
    top_idx = top_idx[np.argsort(-flat[top_idx])] if top_idx.size else top_idx

    return {
        "perPixelSensitivity": flat.tolist(),
        "topSensitivePixels": [
            {
                "index": int(i),
                "row": int((int(i) % (h * w)) // w),
                "col": int(int(i) % w),
                "sensitivity": float(flat[int(i)]),
                "direction": "increase" if float(flat_signed[int(i)]) > 0 else "decrease",
            }
            for i in top_idx
        ],
        "overallSensitivity": float(np.mean(flat)) if flat.size else 0.0,
    }


@router.post("/counterfactual")
def counterfactual(req: CounterfactualRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x0 = tf.convert_to_tensor(_prepare_input(req.originalPixels, arch, req.dataset))
    x1 = tf.convert_to_tensor(_prepare_input(req.modifiedPixels, arch, req.dataset))

    p0 = _binary_probs(tf.cast(model(x0, training=False), tf.float32), req.dataset).numpy()[0]
    p1 = _binary_probs(tf.cast(model(x1, training=False), tf.float32), req.dataset).numpy()[0]
    y0 = int(np.argmax(p0))
    y1 = int(np.argmax(p1))

    delta = np.asarray(req.modifiedPixels, dtype=np.float32) - np.asarray(req.originalPixels, dtype=np.float32)
    return {
        "originalPrediction": {
            "label": _label_for_dataset(req.dataset, y0),
            "confidence": float(np.max(p0) * 100.0),
            "probs": p0.tolist(),
        },
        "modifiedPrediction": {
            "label": _label_for_dataset(req.dataset, y1),
            "confidence": float(np.max(p1) * 100.0),
            "probs": p1.tolist(),
        },
        "predictionFlipped": bool(y0 != y1),
        "confidenceChange": float((np.max(p1) - np.max(p0)) * 100.0),
        "perturbationMagnitude": float(np.linalg.norm(delta)),
        "affectedPixelCount": int(np.sum(np.abs(delta) > 0.01)),
    }


@router.post("/minimal-flip")
def minimal_flip(req: MinimalFlipRequest):
    arch = _arch_key(req.architecture)
    model = _bundle(arch, use_untrained=False).model
    x = tf.convert_to_tensor(_prepare_input(req.pixels, arch, req.dataset))

    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = _binary_probs(tf.cast(model(x, training=False), tf.float32), req.dataset)
        orig_label = int(tf.argmax(probs[0]).numpy())
        loss = -tf.math.log(probs[0, orig_label] + 1e-7)
    grad = tape.gradient(loss, x)
    if grad is None:
        grad = tf.zeros_like(x)
    sign = tf.sign(grad)

    lo, hi = 0.0, 1.0
    best_eps = None
    best = None

    for _ in range(20):
        mid = (lo + hi) / 2.0
        x_adv = tf.clip_by_value(x + mid * sign, 0.0, 1.0)
        p_adv = _binary_probs(tf.cast(model(x_adv, training=False), tf.float32), req.dataset).numpy()[0]
        y_adv = int(np.argmax(p_adv))
        if y_adv != orig_label:
            best_eps = mid
            best = (x_adv.numpy(), p_adv, y_adv)
            hi = mid
        else:
            lo = mid

    if best is None or best_eps is None:
        return {"found": False, "message": "Could not flip prediction within epsilon range"}

    adv_np, adv_probs, adv_label = best
    original = x.numpy().reshape(-1)
    modified = adv_np.reshape(-1)
    diff = np.abs(modified - original)
    return {
        "found": True,
        "epsilon": float(best_eps),
        "modifiedPixels": modified.astype(np.float32).tolist(),
        "perturbationMap": diff.astype(np.float32).tolist(),
        "newPrediction": {
            "label": _label_for_dataset(req.dataset, adv_label),
            "confidence": float(np.max(adv_probs) * 100.0),
            "probs": adv_probs.tolist(),
        },
        "affectedPixelCount": int(np.sum(diff > 0.01)),
    }


@router.post("/comparison/run-all")
def comparison_run_all(req: ComparisonRequest):
    results: dict[str, Any] = {}
    for arch_name in ("ANN", "CNN", "RNN"):
        arch = arch_name.lower()
        try:
            bundle = _bundle(arch, use_untrained=False)
        except HTTPException:
            results[arch_name] = None
            continue

        stages = []
        if arch == "ann":
            stages = [
                "input", "preprocessing", "dense1", "relu1", "dense2", "relu2", "dense_output", "softmax", "output"
            ]
        elif arch == "cnn":
            stages = [
                "input", "preprocessing", "conv1", "relu_conv1", "pool1", "conv2", "relu_conv2", "pool2", "flatten",
                "dense1", "relu_dense", "dense_output", "softmax", "output"
            ]
        else:
            stages = ["input", "preprocessing", "lstm_t1", "lstm_tmid", "lstm_tfinal", "dense_output", "softmax", "output"]

        x = _prepare_input(req.pixels, arch, req.dataset)
        start = time.perf_counter()
        out_map, layers = _collect_outputs(bundle.model, bundle.activation_model, x)
        probs = _binary_probs(tf.cast(bundle.model(x, training=False), tf.float32), req.dataset).numpy()[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        activations = {}
        for idx, sid in enumerate(stages):
            in_data, out_data, lname = _pick_stage(sid, arch, x, out_map)
            layer = layers.get(lname) if lname else None
            param_count = int(layer.count_params()) if layer is not None else 0
            act = {
                "input": np.asarray(in_data, dtype=np.float32).reshape(-1).tolist(),
                "output": _dataset_adjust(np.asarray(out_data, dtype=np.float32).reshape(-1), req.dataset).tolist(),
                "input_shape": list(np.asarray(in_data).shape),
                "output_shape": list(np.asarray(out_data).shape),
                "param_count": param_count,
                "compute_time_ms": elapsed_ms / max(1, len(stages)),
            }
            if layer is not None:
                w = layer.get_weights()
                if w:
                    act["weights"] = np.asarray(w[0]).reshape(-1).astype(np.float32).tolist()
                    if len(w) > 1:
                        act["bias"] = np.asarray(w[1]).reshape(-1).astype(np.float32).tolist()
            if arch == "rnn" and sid.startswith("lstm"):
                act["gates"] = _fake_gates(np.asarray(out_data).reshape(-1), idx)
            activations[sid] = act

        label = int(np.argmax(probs))
        results[arch_name] = {
            "activations": activations,
            "prediction": {
                "label": _label_for_dataset(req.dataset, label),
                "confidence": float(np.max(probs) * 100.0),
                "probs": probs.tolist(),
            },
            "totalTimeMs": elapsed_ms,
        }
    return results
