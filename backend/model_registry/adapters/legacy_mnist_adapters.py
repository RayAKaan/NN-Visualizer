"""Adapters for the legacy NN-Visualizer CNN/RNN MNIST checkpoints.

These models were produced by this repository's own training scripts and are
stored under ``models_store/`` (git-ignored).  They keep the original
"predict a drawn MNIST digit" behaviour and their rich per-layer activation
payloads working.  If the weight file is absent the adapter reports
``unavailable`` (no substitute is fabricated).
"""
from __future__ import annotations

import numpy as np

from model_registry.base import ModelAdapter, ModelLoadError
from model_registry.schema import FAMILY_CNN, FAMILY_RNN

DIGIT_LABELS = [str(i) for i in range(10)]


class _LegacyMNISTAdapter(ModelAdapter):
    model_type = ""            # "cnn" | "rnn" (index into config.MODEL_PATHS)
    path_attr = ""             # e.g. "CNN_MODEL_PATH"
    input_kind = ""            # shape variant, mirrors config.MODEL_INPUT_SHAPES

    def __init__(self):
        import config

        self._path = getattr(config, self.path_attr)
        super().__init__()

    def _weights_exist(self):
        import os

        return os.path.exists(self._path)

    def check_available(self) -> tuple[bool, str | None]:
        if not self._weights_exist():
            return False, (
                f"checkpoint not found at {self._path}. Train it or place the .h5 "
                "file under models_store/ (git-ignored)."
            )
        return True, None

    def _describe(self):
        return {
            "id": self.model_id,
            "name": f"NN-Visualizer legacy {self.model_type.upper()} (MNIST)",
            "family": self.family,
            "framework": "TensorFlow / Keras (project-trained)",
            "source": "This repository — train_cnn.py / train_rnn.py",
            "weights": self._path,
            "dataset": "MNIST (10 digits)",
            "input_type": "mnist_pixels",
            "input_shape": list(self.input_kind),
            "num_classes": 10,
            "preprocessing": "28x28 grayscale -> /255 -> model-specific reshape",
            "architecture": self.arch_desc,
            "parameter_count": None,
            "description": self.desc,
            "license": "MIT",
            "pretrained": False,
        }

    def refresh_metadata(self):
        try:
            if hasattr(self, "_model"):
                self.metadata["parameter_count"] = int(self._model.count_params())
        except Exception:
            pass

    def load(self):
        if not self._weights_exist():
            raise ModelLoadError(
                f"{self.model_id} checkpoint not found at {self._path}. Train it "
                "(backend/train_cnn.py or train_rnn.py) or place the .h5 file in models_store/."
            )
        # Delegate to the existing inference engine (already builds activation
        # models + activation payloads); identical file, no retraining.
        from services.inference import inference_engine

        if self.model_type not in inference_engine.models:
            inference_engine._load_model(self.model_type, self._path)
        self._engine = inference_engine
        return inference_engine.models[self.model_type]

    def predict(self, raw_input):
        pixels = raw_input.get("pixels")
        if not pixels:
            raise ModelLoadError("'pixels' is required for MNIST models")
        engine = getattr(self, "_engine", None)
        if engine is None or self.model_type not in engine.models:
            raise ModelLoadError(f"{self.model_id} is not loaded")
        raw = engine.predict(pixels, self.model_type)
        probs = np.asarray(raw["probabilities"], dtype=np.float64)
        idx = int(np.argmax(probs))
        details = {k: v for k, v in raw.items() if k not in ("prediction", "confidence", "probabilities")}
        return idx, float(probs[idx]), probs.tolist(), DIGIT_LABELS, details


class LegacyCNNAdapter(_LegacyMNISTAdapter):
    model_id = "cnn-nn-visualizer"
    family = FAMILY_CNN
    model_type = "cnn"
    path_attr = "CNN_MODEL_PATH"
    input_kind = (28, 28, 1)
    arch_desc = "Conv2D(32)-Pool-Conv2D(64)-Pool-Flatten-Dense(128)-Dense(10) softmax"
    desc = "Original project-trained MNIST CNN (kept for legacy digit predictions)."


class LegacyRNNAdapter(_LegacyMNISTAdapter):
    model_id = "rnn-nn-visualizer"
    family = FAMILY_RNN
    model_type = "rnn"
    path_attr = "RNN_MODEL_PATH"
    input_kind = (28, 28)
    arch_desc = "LSTM(128) over 28 timesteps of 28 features -> Dense(10) softmax"
    desc = "Original project-trained MNIST row-wise LSTM (kept for legacy digit predictions)."
