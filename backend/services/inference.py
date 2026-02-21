from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import tensorflow as tf

from config import DEFAULT_MODEL_TYPE, MODEL_PATH, SAMPLES_PATH
from model_utils import build_activation_model, extract_weights, load_model, normalize_pixels

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None
        self.activation_model: tf.keras.Model | None = None
        self.model_type = DEFAULT_MODEL_TYPE
        self._weights: dict[str, list[list[float]]] = {}
        self._loaded = False
        self._last_error = "Model not loaded"
        self._samples: dict[str, list[float]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None and self.activation_model is not None

    def load(self, path: str | None = None) -> None:
        try:
            self.model = load_model(path or str(MODEL_PATH))
            self.model.predict(np.zeros((1, 784), dtype=np.float32), verbose=0)
            self.activation_model = build_activation_model(self.model)
            _, w_h1_h2, w_h2_out = extract_weights(self.model)
            self._weights = {
                "hidden1_hidden2": w_h1_h2.tolist(),
                "hidden2_output": w_h2_out.tolist(),
            }
            self._loaded = True
            self._last_error = ""
            try:
                self._load_samples()
            except Exception:  # noqa: BLE001
                logger.warning("Unable to refresh MNIST samples; using fallback samples")
                self._samples = self._fallback_samples()
            logger.info("Inference model loaded successfully from %s", path or MODEL_PATH)
        except Exception as exc:  # noqa: BLE001
            self._loaded = False
            self.model = None
            self.activation_model = None
            self._weights = {}
            self._last_error = str(exc)
            logger.exception("Failed to load inference model")

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(self._last_error or "Model unavailable")

    def _forward(self, pixels: list[float]) -> dict[str, Any]:
        self.ensure_loaded()
        x = normalize_pixels(pixels)
        assert self.activation_model is not None
        activations = self.activation_model.predict(x, verbose=0)
        probs = activations[-1][0].astype(np.float32)
        prediction = int(np.argmax(probs))
        return {
            "input": x[0].tolist(),
            "prediction": prediction,
            "probabilities": probs.tolist(),
            "activations": {
                "hidden1": activations[0][0].tolist(),
                "hidden2": activations[1][0].tolist(),
                "output": probs.tolist(),
            },
        }

    def predict(self, pixels: list[float]) -> dict[str, Any]:
        result = self._forward(pixels)
        return {
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
            "layers": {
                "hidden1": result["activations"]["hidden1"],
                "hidden2": result["activations"]["hidden2"],
            },
            "activations": result["activations"],
            "input": result["input"],
        }

    def _build_edges(self, matrix: list[list[float]], source: list[float], top_k: int = 200) -> list[dict[str, Any]]:
        arr_w = np.array(matrix, dtype=np.float32)
        src = np.array(source, dtype=np.float32)
        contrib = arr_w * src[:, None]
        if contrib.size == 0:
            return []
        flat = [
            (i, j, float(contrib[i, j]))
            for i in range(contrib.shape[0])
            for j in range(contrib.shape[1])
        ]
        flat.sort(key=lambda item: abs(item[2]), reverse=True)
        return [
            {"from": int(i), "to": int(j), "strength": float(v)}
            for i, j, v in flat[:top_k]
        ]

    def get_state(self, pixels: list[float]) -> dict[str, Any]:
        result = self._forward(pixels)
        output = result["activations"]["output"]
        prediction = result["prediction"]
        return {
            "input": result["input"],
            "layers": result["activations"],
            "prediction": prediction,
            "confidence": float(output[prediction]) if output else 0.0,
            "edges": {
                "hidden1_hidden2": self._build_edges(
                    self._weights.get("hidden1_hidden2", []), result["activations"]["hidden1"], 200
                ),
                "hidden2_output": self._build_edges(
                    self._weights.get("hidden2_output", []), result["activations"]["hidden2"], 200
                ),
            },
        }

    def get_weights(self) -> dict[str, list[list[float]]]:
        self.ensure_loaded()
        return self._weights

    def get_model_info(self) -> dict[str, Any]:
        self.ensure_loaded()
        assert self.model is not None
        return {
            "type": self.model_type,
            "layers": [
                {"name": "input", "size": 784},
                {"name": "hidden1", "size": 128, "activation": "relu"},
                {"name": "hidden2", "size": 64, "activation": "relu"},
                {"name": "output", "size": 10, "activation": "softmax"},
            ],
            "total_params": int(self.model.count_params()),
            "accuracy": 0.975,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.is_loaded,
            "model_type": self.model_type,
        }

    def _load_samples(self) -> None:
        if SAMPLES_PATH.exists():
            with SAMPLES_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    self._samples = {k: v for k, v in data.items() if isinstance(v, list) and len(v) == 784}
                    return
        try:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x_train = x_train.astype("float32") / 255.0
            samples: dict[str, list[float]] = {}
            for digit in range(10):
                idx = int(np.where(y_train == digit)[0][0])
                samples[str(digit)] = x_train[idx].reshape(-1).tolist()
            self._samples = samples
            with SAMPLES_PATH.open("w", encoding="utf-8") as fh:
                json.dump(samples, fh)
        except Exception:  # noqa: BLE001
            self._samples = self._fallback_samples()

    def _fallback_samples(self) -> dict[str, list[float]]:
        samples: dict[str, list[float]] = {}
        for digit in range(10):
            arr = np.zeros((28, 28), dtype=np.float32)
            arr[4:24, 10 + (digit % 3):12 + (digit % 3)] = 1.0
            samples[str(digit)] = arr.reshape(-1).tolist()
        return samples

    def get_samples(self) -> dict[str, list[float]]:
        if not self._samples:
            self._load_samples()
        return self._samples


inference_engine = InferenceEngine()
