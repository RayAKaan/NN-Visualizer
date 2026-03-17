from __future__ import annotations

import uuid
from typing import Dict, Tuple

import numpy as np


def _shuffle_split(x: np.ndarray, y: np.ndarray, train_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    cut = int(len(x) * train_split)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def _make_sine(n_samples: int, seq_len: int, n_features: int, n_classes: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        cls = np.random.randint(0, n_classes)
        freq = 1 + cls
        t = np.linspace(0, 2 * np.pi, seq_len)
        signal = np.sin(freq * t)
        if n_features > 1:
            signal = np.stack([signal] * n_features, axis=1)
        else:
            signal = signal[:, None]
        signal += np.random.normal(0.0, noise, signal.shape)
        x[i] = signal
        y[i] = cls
    return x, y


def _make_square(n_samples: int, seq_len: int, n_features: int, n_classes: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)
    for i in range(n_samples):
        cls = np.random.randint(0, n_classes)
        freq = 1 + cls
        t = np.linspace(0, 2 * np.pi, seq_len)
        signal = np.sign(np.sin(freq * t))
        if n_features > 1:
            signal = np.stack([signal] * n_features, axis=1)
        else:
            signal = signal[:, None]
        signal += np.random.normal(0.0, noise, signal.shape)
        x[i] = signal
        y[i] = cls
    return x, y


def _make_noise(n_samples: int, seq_len: int, n_features: int, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.normal(0.0, 1.0, size=(n_samples, seq_len, n_features)).astype(np.float32)
    y = np.random.randint(0, n_classes, size=(n_samples,), dtype=np.int64)
    return x, y


def generate_sequence_dataset(
    seq_type: str,
    n_samples: int,
    seq_len: int,
    n_features: int,
    n_classes: int,
    noise: float,
    train_split: float,
) -> Dict:
    n_samples = max(10, int(n_samples))
    seq_len = max(2, int(seq_len))
    n_features = max(1, int(n_features))
    n_classes = max(2, int(n_classes))
    noise = float(noise)
    train_split = float(train_split)

    dtype = (seq_type or "sine_wave").lower()
    if dtype in {"sine", "sine_wave"}:
        x, y = _make_sine(n_samples, seq_len, n_features, n_classes, noise)
    elif dtype in {"square", "square_wave"}:
        x, y = _make_square(n_samples, seq_len, n_features, n_classes, noise)
    elif dtype == "noise":
        x, y = _make_noise(n_samples, seq_len, n_features, n_classes)
    else:
        raise ValueError("Unknown sequence type.")

    x_train, y_train, x_test, y_test = _shuffle_split(x, y, train_split)
    return {
        "dataset_id": str(uuid.uuid4()),
        "input_shape": [seq_len, n_features],
        "output_shape": [n_classes],
        "data_type": "sequence",
        "train": {
            "x": x_train.tolist(),
            "y": _one_hot(y_train, n_classes).tolist(),
        },
        "test": {
            "x": x_test.tolist(),
            "y": _one_hot(y_test, n_classes).tolist(),
        },
    }

