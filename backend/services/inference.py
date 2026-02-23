from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from config import (
    CNN_MODEL_PATH,
    CNN_MODEL_PATH_H5,
    DEFAULT_MODEL_TYPE,
    MODEL_PATH,
    SAMPLES_PATH,
    SUPPORTED_MODEL_TYPES,
)
from model_utils import normalize_pixels

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self) -> None:
        self.models: dict[str, tf.keras.Model] = {}
        self.activation_models: dict[str, tf.keras.Model] = {}
        self.weights_cache: dict[str, dict[str, Any]] = {}
        self.active_model_type = DEFAULT_MODEL_TYPE
        self.loaded = False
        self._last_error = "Models not loaded"
        self._samples: dict[str, list[float]] = {}

    @property
    def is_loaded(self) -> bool:
        return self.loaded and bool(self.models)

    def load_all(self) -> None:
        self.models = {}
        self.activation_models = {}
        self.weights_cache = {}

        ann_path = Path(MODEL_PATH)
        if ann_path.exists():
            self._load_model("ann", str(ann_path))

        cnn_path = Path(CNN_MODEL_PATH)
        if not cnn_path.exists():
            cnn_path = Path(CNN_MODEL_PATH_H5)
        if cnn_path.exists():
            self._load_model("cnn", str(cnn_path))
        else:
            logger.warning("CNN model file not found at %s or %s", CNN_MODEL_PATH, CNN_MODEL_PATH_H5)

        self.loaded = bool(self.models)
        if not self.loaded:
            self._last_error = "No models available"

        if "ann" in self.models:
            self.active_model_type = "ann"
        elif self.models:
            self.active_model_type = list(self.models.keys())[0]

        self._load_samples()

    def load(self, path: str) -> None:
        # Backward-compatible single-model loader for saved ANN models.
        self._load_model("ann", path)
        self.active_model_type = "ann"
        self.loaded = True

    def _load_model(self, model_type: str, path: str) -> None:
        model = tf.keras.models.load_model(path)

        if model_type == "ann":
            model.predict(np.zeros((1, 784), dtype=np.float32), verbose=0)
        elif model_type == "cnn":
            model.predict(np.zeros((1, 28, 28, 1), dtype=np.float32), verbose=0)

        layer_outputs = [
            layer.output
            for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]
        activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

        self.models[model_type] = model
        self.activation_models[model_type] = activation_model
        self._cache_weights(model_type)
        logger.info("Loaded %s model from %s", model_type, path)

    def _cache_weights(self, model_type: str) -> None:
        model = self.models[model_type]
        weights: dict[str, Any] = {}
        for layer in model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                weights[layer.name] = {
                    "kernel": w[0].tolist(),
                    "bias": w[1].tolist() if len(w) > 1 else None,
                }

        if model_type == "ann":
            raw = model.get_weights()
            weights["hidden1_hidden2"] = raw[2].tolist()
            weights["hidden2_output"] = raw[4].tolist()

        self.weights_cache[model_type] = weights

    def ensure_available(self, model_type: str) -> None:
        if model_type not in self.models:
            raise RuntimeError(f"Model '{model_type}' not loaded")

    def set_active_model(self, model_type: str) -> None:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unknown model type '{model_type}'")
        self.ensure_available(model_type)
        self.active_model_type = model_type

    def get_available_models(self) -> list[str]:
        return list(self.models.keys())

    def predict_ann(self, pixels: list[float]) -> dict[str, Any]:
        self.ensure_available("ann")
        x = normalize_pixels(pixels)
        model = self.models["ann"]
        act_model = self.activation_models["ann"]
        activations = act_model.predict(x, verbose=0)

        non_input_layers = [
            layer
            for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]
        layers_out: dict[str, list[float]] = {}
        for layer, act in zip(non_input_layers, activations):
            layers_out[layer.name] = act[0].tolist()

        hidden1 = next((vals for vals in layers_out.values() if len(vals) == 128), [])
        hidden2 = next((vals for vals in layers_out.values() if len(vals) == 64), [])
        output = next((vals for vals in layers_out.values() if len(vals) == 10), [])

        prediction = int(np.argmax(output)) if output else -1
        confidence = float(max(output)) if output else 0.0

        return {
            "model_type": "ann",
            "input": x[0].tolist(),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": output,
            "layers": {
                "hidden1": hidden1,
                "hidden2": hidden2,
            },
            "activations": {
                "hidden1": hidden1,
                "hidden2": hidden2,
                "output": output,
            },
        }

    def predict_cnn(self, pixels: list[float]) -> dict[str, Any]:
        self.ensure_available("cnn")
        x = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1)
        act_model = self.activation_models["cnn"]
        model = self.models["cnn"]
        activations = act_model.predict(x, verbose=0)
        non_input_layers = [
            layer
            for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)
        ]

        feature_maps: list[dict[str, Any]] = []
        dense_layers: dict[str, list[float]] = {}
        layer_activations: dict[str, Any] = {}
        output_probs: list[float] = []

        for layer, act in zip(non_input_layers, activations):
            name = layer.name
            shape = act.shape
            if len(shape) == 4:
                _, h, w, c = shape
                act_squeezed = act[0]
                mean_per_filter = np.mean(act_squeezed, axis=(0, 1))
                ranking = np.argsort(-mean_per_filter).tolist()
                top_k = min(16, c)
                top_indices = ranking[:top_k]
                fm_data = {str(idx): act_squeezed[:, :, idx].tolist() for idx in top_indices}
                feature_maps.append(
                    {
                        "layer_name": name,
                        "layer_type": "conv" if "conv" in name else "pool",
                        "shape": [int(h), int(w), int(c)],
                        "top_k": top_k,
                        "activation_ranking": top_indices,
                        "feature_maps": fm_data,
                        "mean_activations": mean_per_filter.tolist(),
                    }
                )
                layer_activations[name] = {
                    "type": "spatial",
                    "shape": [int(h), int(w), int(c)],
                    "global_mean": float(np.mean(act_squeezed)),
                }
            elif len(shape) == 2:
                vals = act[0].tolist()
                layer_activations[name] = {"type": "dense", "values": vals}
                if len(vals) == 10:
                    output_probs = vals
                else:
                    dense_layers[name] = vals

        prediction = int(np.argmax(output_probs)) if output_probs else -1
        confidence = float(max(output_probs)) if output_probs else 0.0

        kernels: list[dict[str, Any]] = []
        for fm in feature_maps:
            lname = fm["layer_name"]
            if "conv" not in lname:
                continue
            layer = model.get_layer(lname)
            w = layer.get_weights()
            if len(w) == 0:
                continue
            kernel = w[0]
            kernel_data = {}
            for idx in fm["activation_ranking"]:
                kernel_data[str(idx)] = kernel[:, :, 0, idx].tolist()
            kernels.append(
                {
                    "layer_name": lname,
                    "kernel_shape": list(kernel.shape),
                    "kernels": kernel_data,
                }
            )

        return {
            "model_type": "cnn",
            "input": np.array(pixels, dtype=np.float32).tolist(),
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": output_probs,
            "feature_maps": feature_maps,
            "kernels": kernels,
            "dense_layers": dense_layers,
            "layer_activations": layer_activations,
        }

    def predict(self, pixels: list[float], model_type: str | None = None) -> dict[str, Any]:
        mt = model_type or self.active_model_type
        if mt == "ann":
            return self.predict_ann(pixels)
        if mt == "cnn":
            return self.predict_cnn(pixels)
        raise ValueError(f"Unknown model type: {mt}")

    def _build_edges(self, matrix: np.ndarray, source: list[float], top_k: int = 200) -> list[dict[str, Any]]:
        src = np.array(source, dtype=np.float32)
        contrib = matrix * src[:, None]
        flat = [
            (i, j, float(contrib[i, j]))
            for i in range(contrib.shape[0])
            for j in range(contrib.shape[1])
        ]
        flat.sort(key=lambda item: abs(item[2]), reverse=True)
        return [{"from": int(i), "to": int(j), "strength": float(v)} for i, j, v in flat[:top_k]]

    def get_state_ann(self, pixels: list[float]) -> dict[str, Any]:
        result = self.predict_ann(pixels)
        w_h1_h2 = np.array(self.weights_cache["ann"].get("hidden1_hidden2", []), dtype=np.float32)
        w_h2_out = np.array(self.weights_cache["ann"].get("hidden2_output", []), dtype=np.float32)
        return {
            "model_type": "ann",
            "input": result["input"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "layers": {
                "hidden1": result["layers"]["hidden1"],
                "hidden2": result["layers"]["hidden2"],
            },
            "edges": {
                "hidden1_hidden2": self._build_edges(w_h1_h2, result["layers"]["hidden1"], 200),
                "hidden2_output": self._build_edges(w_h2_out, result["layers"]["hidden2"], 200),
            },
        }

    def get_state_cnn(self, pixels: list[float]) -> dict[str, Any]:
        result = self.predict_cnn(pixels)
        model = self.models["cnn"]

        dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        dense_weights: dict[str, np.ndarray] = {}
        for layer in dense_layers:
            w = layer.get_weights()
            if len(w) > 0:
                dense_weights[layer.name] = w[0]

        dense_vec = result["dense_layers"].get("dense", [])
        output_probs = result.get("probabilities", [])

        hidden1 = dense_vec
        hidden2 = dense_vec[:64] if len(dense_vec) >= 64 else dense_vec

        w_dense_out = dense_weights.get("dense_1")
        edges_hidden2_output: list[dict[str, Any]] = []
        if w_dense_out is not None:
            use_len = min(len(hidden2), w_dense_out.shape[0])
            for si in range(use_len):
                for di in range(w_dense_out.shape[1]):
                    strength = float(w_dense_out[si][di] * hidden2[si])
                    edges_hidden2_output.append({"from": si, "to": di, "strength": strength})
            edges_hidden2_output.sort(key=lambda e: abs(e["strength"]), reverse=True)
            edges_hidden2_output = edges_hidden2_output[:200]

        return {
            **result,
            "layers": {
                "hidden1": hidden1,
                "hidden2": hidden2,
                "output": output_probs,
            },
            "edges": {
                "hidden1_hidden2": [],
                "hidden2_output": edges_hidden2_output,
            },
        }

    def get_state(self, pixels: list[float], model_type: str | None = None) -> dict[str, Any]:
        mt = model_type or self.active_model_type
        if mt == "ann":
            return self.get_state_ann(pixels)
        if mt == "cnn":
            return self.get_state_cnn(pixels)
        raise ValueError(f"Unknown model type: {mt}")

    def get_weights(self, model_type: str | None = None) -> dict[str, Any]:
        mt = model_type or self.active_model_type
        self.ensure_available(mt)
        return self.weights_cache.get(mt, {})

    def get_model_info(self, model_type: str | None = None) -> dict[str, Any]:
        mt = model_type or self.active_model_type
        self.ensure_available(mt)
        model = self.models[mt]
        layers_info = []
        for layer in model.layers:
            info: dict[str, Any] = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "params": layer.count_params(),
            }
            cfg = layer.get_config()
            for key in ["activation", "filters", "kernel_size", "pool_size", "units", "rate"]:
                if key in cfg:
                    info[key] = cfg[key]
            try:
                info["output_shape"] = str(layer.output_shape)
            except Exception:  # noqa: BLE001
                pass
            layers_info.append(info)

        return {
            "type": mt,
            "layers": layers_info,
            "total_params": int(model.count_params()),
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "accuracy": 0.975 if mt == "ann" else 0.99,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "models_loaded": {name: (name in self.models) for name in SUPPORTED_MODEL_TYPES},
            "active_model": self.active_model_type,
        }

    def _load_samples(self) -> None:
        if SAMPLES_PATH.exists():
            with SAMPLES_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    self._samples = {
                        k: v for k, v in data.items() if isinstance(v, list) and len(v) == 784
                    }
                    if self._samples:
                        return
        self._samples = self._fallback_samples()

    def _fallback_samples(self) -> dict[str, list[float]]:
        samples: dict[str, list[float]] = {}
        for digit in range(10):
            arr = np.zeros((28, 28), dtype=np.float32)
            arr[4:24, 10 + (digit % 3):12 + (digit % 3)] = 1.0
            samples[str(digit)] = arr.reshape(-1).tolist()
        return samples

    def get_samples(self, model_type: str | None = None) -> dict[str, list[float]]:
        _ = model_type
        if not self._samples:
            self._load_samples()
        return self._samples


inference_engine = InferenceEngine()
