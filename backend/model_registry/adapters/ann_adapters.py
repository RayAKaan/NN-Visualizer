"""Adapters for the ANN (MLP / MNIST) family."""
from __future__ import annotations

import numpy as np

from model_registry import store
from model_registry.base import ModelAdapter, ModelLoadError
from model_registry.safetensors import load_safetensors
from model_registry.schema import FAMILY_ANN
from model_registry.util import decode_pixels, normalize_probs, softmax

DIGIT_LABELS = [str(i) for i in range(10)]


# ===========================================================================
# dacorvo/mnist-mlp — official PyTorch MLP checkpoint (NumPy inference)
# ===========================================================================
class DacorvoMnistMlpAdapter(ModelAdapter):
    model_id = "ann-dacorvo-mnist-mlp"
    HF_REPO = "dacorvo/mnist-mlp"
    HF_FILE = "model.safetensors"
    WEIGHTS_URL = store.hf_url(HF_REPO, HF_FILE)

    def _describe(self):
        return {
            "id": self.model_id,
            "name": "dacorvo MNIST MLP",
            "family": FAMILY_ANN,
            "framework": "PyTorch (official weights) / NumPy inference",
            "source": "Hugging Face — dacorvo/mnist-mlp",
            "weights": self.WEIGHTS_URL,
            "dataset": "MNIST (10 digits)",
            "input_type": "mnist_pixels",
            "input_shape": [784],
            "num_classes": 10,
            "preprocessing": "28x28 grayscale -> flatten -> (x - 0.1307) / 0.3081",
            "architecture": "MLP 784-256-256-10 · ReLU hidden layers, softmax output",
            "parameter_count": 269_322,
            "description": "Reference MLP for quantization studies, ~269K parameters.",
            "license": "Apache-2.0",
            "pretrained": True,
        }

    def load(self):
        path = store.ensure_downloaded(self.WEIGHTS_URL, "dacorvo_mnist_mlp.safetensors")
        st = load_safetensors(path)
        # PyTorch Linear stores weights as [out, in]; forward uses x @ W.T
        self._w = {
            "w1": st["input_layer.weight"].astype(np.float32),
            "b1": st["input_layer.bias"].astype(np.float32),
            "w2": st["mid_layer.weight"].astype(np.float32),
            "b2": st["mid_layer.bias"].astype(np.float32),
            "w3": st["output_layer.weight"].astype(np.float32),
            "b3": st["output_layer.bias"].astype(np.float32),
        }
        store.write_provenance(
            self.model_id,
            {"repo": self.HF_REPO, "file": self.HF_FILE, "license": "apache-2.0"},
        )
        return self._w

    def predict(self, raw_input):
        pixels = decode_pixels(raw_input.get("pixels"))
        if not hasattr(self, "_w"):
            raise ModelLoadError(f"{self.model_id} is not loaded")
        x = (pixels - 0.1307) / 0.3081
        h1 = np.maximum(0.0, x @ self._w["w1"].T + self._w["b1"])
        h2 = np.maximum(0.0, h1 @ self._w["w2"].T + self._w["b2"])
        logits = h2 @ self._w["w3"].T + self._w["b3"]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs.tolist(), DIGIT_LABELS, None


# ===========================================================================
# mysticdan/mlp-mnist — official JAX MLP checkpoint (NumPy inference)
# ===========================================================================
class MysticdanMnistMlpAdapter(ModelAdapter):
    model_id = "ann-mysticdan-mlp-mnist"
    HF_REPO = "mysticdan/mlp-mnist"
    HF_FILE = "mlp_mnist_model.pkl"
    WEIGHTS_URL = store.hf_url(HF_REPO, HF_FILE)
    LAYERS = [784, 512, 256, 128, 64, 10]  # verified from the checkpoint

    def _describe(self):
        return {
            "id": self.model_id,
            "name": "mysticdan MNIST MLP (JAX)",
            "family": FAMILY_ANN,
            "framework": "JAX (official weights) / NumPy inference",
            "source": "Hugging Face — mysticdan/mlp-mnist",
            "weights": self.WEIGHTS_URL,
            "dataset": "MNIST (10 digits)",
            "input_type": "mnist_pixels",
            "input_shape": [784],
            "num_classes": 10,
            "preprocessing": "28x28 grayscale -> /255 -> flatten (no mean/std)",
            "architecture": "MLP 784-512-256-128-64-10 · ReLU hidden layers, logits->softmax",
            "parameter_count": 575_050,
            "description": "From-scratch JAX MLP (He init, ~98% MNIST test accuracy).",
            "license": "MIT",
            "pretrained": True,
        }

    # -- weights ---------------------------------------------------------
    def _npz_path(self):
        return store.cache_path("mysticdan_mnist_mlp.npz")

    def load(self):
        npz = self._npz_path()
        if not (npz and __import__("os").path.exists(npz)):
            pkl = store.ensure_downloaded(self.WEIGHTS_URL, "mysticdan_mlp_mnist.pkl")
            _convert_pkl_to_npz(pkl, npz)
        d = np.load(npz)
        self._params = [(d[f"w{i}"], d[f"b{i}"]) for i in range(len(self.LAYERS) - 1)]
        store.write_provenance(
            self.model_id,
            {"repo": self.HF_REPO, "file": self.HF_FILE, "license": "mit"},
        )
        return self._params

    def predict(self, raw_input):
        pixels = decode_pixels(raw_input.get("pixels"))
        if not hasattr(self, "_params"):
            raise ModelLoadError(f"{self.model_id} is not loaded")
        h = pixels
        n = len(self._params)
        for i, (w, b) in enumerate(self._params):
            h = h @ w + b
            if i < n - 1:
                h = np.maximum(0.0, h)  # ReLU (verified activation map)
        probs = softmax(h)
        idx = int(np.argmax(probs))
        return idx, float(probs[idx]), probs.tolist(), DIGIT_LABELS, None


def _convert_pkl_to_npz(pkl_path: str, npz_path: str) -> None:
    """Convert the JAX pickle into a plain NumPy archive.

    Prefers an in-process ``jax`` import when available, otherwise shells out
    to a fresh interpreter that uses a small pickle-reconstruction shim so the
    conversion works without JAX installed.
    """
    try:
        import jax  # noqa: F401  (ensures native unpickle works)

        def _load():
            import pickle

            with open(pkl_path, "rb") as fh:
                return pickle.load(fh)

    except Exception:  # JAX unavailable -> subprocess shim
        import subprocess
        import sys

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "model_registry.adapters._mysticdan_convert",
                pkl_path,
                npz_path,
            ]
        )
        return

    import os
    import pickle

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    params = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in data["params"]]
    out = {f"w{i}": w for i, (w, _) in enumerate(params)}
    out.update({f"b{i}": b for i, (_, b) in enumerate(params)})
    np.savez(npz_path, **out)


# ===========================================================================
# NN-Visualizer existing ANN baseline (backend/nn_visual_ann.h5)
# ===========================================================================
class NNVisualizerAnnAdapter(ModelAdapter):
    model_id = "ann-nn-visualizer"

    def __init__(self):
        import config

        self._path = config.ANN_MODEL_PATH
        super().__init__()

    def _describe(self):
        return {
            "id": self.model_id,
            "name": "NN-Visualizer ANN (existing baseline)",
            "family": FAMILY_ANN,
            "framework": "TensorFlow / Keras",
            "source": "This repository (backend/nn_visual_ann.h5)",
            "weights": self._path,
            "dataset": "MNIST (10 digits)",
            "input_type": "mnist_pixels",
            "input_shape": [784],
            "num_classes": 10,
            "preprocessing": "28x28 grayscale -> /255 -> flatten",
            "architecture": "MLP 784-256-128-64-10 · ReLU/Dropout, softmax output",
            "parameter_count": 242_762,
            "description": "Project-trained MNIST ANN used by the original Prediction Mode.",
            "license": "MIT",
            "pretrained": True,
        }

    def check_available(self) -> tuple[bool, str | None]:
        import os

        if not os.path.exists(self._path):
            return False, (
                f"checkpoint not found at {self._path}. Add nn_visual_ann.h5 under "
                "models_store/ or backend/ (see README)."
            )
        return True, None

    def load(self):
        import os

        import tensorflow as tf

        if not os.path.exists(self._path):
            raise ModelLoadError(
                f"ANN checkpoint not found at {self._path}. Add nn_visual_ann.h5 under "
                "models_store/ or backend/ (see README)."
            )
        self._model = tf.keras.models.load_model(self._path)
        return self._model

    def predict(self, raw_input):
        pixels = decode_pixels(raw_input.get("pixels"))
        if not hasattr(self, "_model"):
            raise ModelLoadError(f"{self.model_id} is not loaded")
        # frontend pixel arrays are already normalized to [0, 1] (the legacy
        # prediction pipeline never divides by 255 again)
        x = pixels.reshape(1, 784)
        probs = np.asarray(self._model.predict(x, verbose=0)[0], dtype=np.float64)
        probs = normalize_probs(probs)
        idx = int(np.argmax(probs))
        extra = {"layers": _legacy_ann_layers(self._model, x, probs)}
        return idx, float(probs[idx]), probs.tolist(), DIGIT_LABELS, extra


def _legacy_ann_layers(model, x, probs):
    """Recompute hidden activations for the legacy layer-by-layer trace UI."""
    try:
        import tensorflow as tf

        tracked = [l.output for l in model.layers if l.name in ("hidden1", "hidden2", "hidden3")]
        if not tracked:
            return {}
        act_model = tf.keras.Model(inputs=model.input, outputs=tracked)
        outputs = act_model.predict(x, verbose=0)
        layers = {}
        for i, name in enumerate(("hidden1", "hidden2", "hidden3")):
            if i < len(outputs):
                layers[name] = np.asarray(outputs[i][0], dtype=np.float64).tolist()
        return layers
    except Exception:  # viz is best-effort
        return {}
