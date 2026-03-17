from __future__ import annotations

import os
import uuid
from typing import Dict, Tuple

import numpy as np


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _preview_images(x: np.ndarray, n: int = 20) -> list:
    preview = []
    for i in range(min(n, len(x))):
        preview.append({"index": int(i), "image": x[i].reshape(-1).tolist(), "label": int(0), "image_base64": ""})
    return preview


def load_cifar10(n_samples: int, train_split: float, seed: int | None = None) -> Dict:
    if seed is not None:
        np.random.seed(seed)
    n_samples = max(50, int(n_samples))
    train_split = float(train_split)
    data_path = os.path.join("models_store", "cifar10_small.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError("CIFAR-10 data not found. Place cifar10_small.npz in models_store.")
    x_train, y_train, x_test, y_test = _load_npz(data_path)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    idx = np.random.permutation(len(x))[:n_samples]
    x = x[idx]
    y = y[idx]
    cut = int(len(x) * train_split)
    x_train, y_train = x[:cut], y[:cut]
    x_test, y_test = x[cut:], y[cut:]
    return {
        "dataset_id": str(uuid.uuid4()),
        "input_shape": [1, 8, 8],
        "output_shape": [10],
        "n_classes": 10,
        "class_names": [str(i) for i in range(10)],
        "train": {
            "n_samples": int(len(x_train)),
            "class_distribution": {str(i): int(np.sum(y_train == i)) for i in range(10)},
            "samples_preview": _preview_images(x_train),
            "x": x_train.reshape(len(x_train), 1, 8, 8).tolist(),
            "y": _one_hot(y_train, 10).tolist(),
        },
        "test": {
            "n_samples": int(len(x_test)),
            "class_distribution": {str(i): int(np.sum(y_test == i)) for i in range(10)},
            "x": x_test.reshape(len(x_test), 1, 8, 8).tolist(),
            "y": _one_hot(y_test, 10).tolist(),
        },
    }
