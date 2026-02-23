import hashlib
import os
from collections import OrderedDict
from typing import Any

import numpy as np
import tensorflow as tf

import config


class InferenceCache:
    def __init__(self, maxsize: int = 32):
        self.maxsize = maxsize
        self.data: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _key(self, pixels: list[float], model_type: str) -> str:
        raw = (model_type + ":" + ",".join(f"{p:.4f}" for p in pixels)).encode()
        return hashlib.md5(raw).hexdigest()

    def get(self, pixels: list[float], model_type: str):
        key = self._key(pixels, model_type)
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def put(self, pixels: list[float], model_type: str, value: dict[str, Any]):
        key = self._key(pixels, model_type)
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.maxsize:
            self.data.popitem(last=False)

    def clear(self):
        self.data.clear()


class InferenceEngine:
    def __init__(self):
        self.models = {}
        self.activation_models = {}
        self.weights_cache = {}
        self.active_model_type = "ann"
        self.loaded = False
        self._cache = InferenceCache(maxsize=32)

    def load_all(self):
        for model_type, path in config.MODEL_PATHS.items():
            if os.path.exists(path):
                self._load_model(model_type, path)
                print(f"Loaded {model_type} model from {path}")
            else:
                print(f"WARNING: {model_type} model not found at {path}")
        if self.models:
            self.active_model_type = "ann" if "ann" in self.models else list(self.models.keys())[0]
            self.loaded = True

    def _load_model(self, model_type, path):
        model = tf.keras.models.load_model(path)
        dummy = np.zeros((1,) + config.MODEL_INPUT_SHAPES[model_type])
        model.predict(dummy, verbose=0)
        outputs = [l.output for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
        act_model = tf.keras.Model(inputs=model.input, outputs=outputs)
        self.models[model_type] = model
        self.activation_models[model_type] = act_model
        self._cache_weights(model_type)

    def _cache_weights(self, model_type: str):
        model = self.models[model_type]
        summary = {}
        for layer in model.layers:
            w = layer.get_weights()
            if w:
                summary[layer.name] = [arr.tolist() for arr in w]
        self.weights_cache[model_type] = summary

    def predict(self, pixels, model_type=None):
        mt = model_type or self.active_model_type
        cached = self._cache.get(pixels, mt)
        if cached:
            return cached
        if mt == "ann":
            result = self._predict_ann(pixels)
        elif mt == "cnn":
            result = self._predict_cnn(pixels)
        elif mt == "rnn":
            result = self._predict_rnn(pixels)
        else:
            raise ValueError(f"Unknown model: {mt}")
        self._cache.put(pixels, mt, result)
        return result

    def _predict_ann(self, pixels):
        x = np.array(pixels, dtype=np.float32).reshape(1, 784)
        outputs = self.activation_models["ann"].predict(x, verbose=0)
        probs = outputs[-1][0]
        return {
            "model_type": "ann",
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": probs.tolist(),
            "layers": {
                "hidden1": outputs[0][0].tolist() if len(outputs) > 1 else [],
                "hidden2": outputs[1][0].tolist() if len(outputs) > 2 else [],
                "hidden3": outputs[2][0].tolist() if len(outputs) > 3 else [],
            },
        }

    def _predict_cnn(self, pixels):
        x = np.array(pixels, dtype=np.float32).reshape(1, 28, 28, 1)
        outputs = self.activation_models["cnn"].predict(x, verbose=0)
        model = self.models["cnn"]
        probs = outputs[-1][0]
        feature_maps = []
        dense_layers = {}
        out_idx = 0
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            act = outputs[out_idx]
            out_idx += 1
            if len(act.shape) == 4:
                fmap = act[0]
                means = fmap.mean(axis=(0, 1))
                ranking = np.argsort(-means)[:16]
                feature_maps.append({
                    "layer_name": layer.name,
                    "layer_type": layer.__class__.__name__,
                    "shape": list(fmap.shape),
                    "top_k": 16,
                    "activation_ranking": ranking.tolist(),
                    "feature_maps": fmap[:, :, ranking].transpose(2, 0, 1).tolist(),
                    "mean_activations": means.tolist(),
                })
            elif len(act.shape) == 2 and layer.name != "output":
                dense_layers[layer.name] = act[0].tolist()
        kernels = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                k = layer.get_weights()[0]
                kernels.append({"layer_name": layer.name, "kernel_shape": list(k.shape), "kernels": k.tolist()})
        return {
            "model_type": "cnn",
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": probs.tolist(),
            "feature_maps": feature_maps,
            "kernels": kernels,
            "dense_layers": dense_layers,
        }

    def _predict_rnn(self, pixels):
        x = np.array(pixels, dtype=np.float32).reshape(1, 28, 28)
        outputs = self.activation_models["rnn"].predict(x, verbose=0)
        probs = outputs[-1][0]
        timestep_activations = np.mean(x[0], axis=1).tolist()
        dense_layers = {}
        model = self.models["rnn"]
        out_idx = 0
        lstm_output = []
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            act = outputs[out_idx]
            out_idx += 1
            if "lstm" in layer.name:
                lstm_output = act[0].tolist()
            elif len(act.shape) == 2 and layer.name != "output":
                dense_layers[layer.name] = act[0].tolist()
        arr = np.array(lstm_output, dtype=np.float32)
        return {
            "model_type": "rnn",
            "prediction": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": probs.tolist(),
            "timestep_activations": timestep_activations,
            "lstm_output": lstm_output,
            "cell_state_summary": {
                "mean": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "min": float(arr.min()) if arr.size else 0.0,
            },
            "dense_layers": dense_layers,
        }

    def get_state(self, pixels, model_type=None):
        result = self.predict(pixels, model_type)
        if result["model_type"] == "ann":
            model = self.models["ann"]
            ann_edges = []
            hidden_layers = [result["layers"]["hidden1"], result["layers"]["hidden2"], result["layers"]["hidden3"]]
            prev = np.array(pixels)
            for idx, lname in enumerate(["hidden1", "hidden2", "hidden3", "output"]):
                weights = model.get_layer(lname).get_weights()[0]
                act = np.array(hidden_layers[idx]) if idx < 3 else np.array(result["probabilities"])
                strength = weights * prev[:, None]
                flat = np.argsort(-np.abs(strength), axis=None)[:200]
                for f in flat:
                    i, j = np.unravel_index(f, strength.shape)
                    ann_edges.append({"from": int(i), "to": int(j), "strength": float(strength[i, j])})
                prev = act
            result["edges"] = ann_edges
        return result

    def get_weights(self, model_type=None):
        mt = model_type or self.active_model_type
        return self.weights_cache.get(mt, {})

    def get_model_info(self, model_type=None):
        mt = model_type or self.active_model_type
        if mt not in self.models:
            return {"model_type": mt, "loaded": False}
        model = self.models[mt]
        layers = []
        for layer in model.layers:
            layers.append(
                {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(getattr(layer, "output_shape", "")),
                    "params": int(layer.count_params()),
                    "activation": getattr(layer, "activation", None).__name__ if getattr(layer, "activation", None) else None,
                }
            )
        return {"model_type": mt, "loaded": True, "total_params": int(model.count_params()), "layers": layers}

    def get_available_models(self):
        return list(self.models.keys())

    def set_active_model(self, model_type: str):
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} is not loaded")
        self.active_model_type = model_type
        self._cache.clear()


inference_engine = InferenceEngine()
