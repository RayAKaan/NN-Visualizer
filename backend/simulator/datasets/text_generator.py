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


def generate_text_tokens(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    n_classes: int,
    train_split: float,
) -> Dict:
    n_samples = max(10, int(n_samples))
    seq_len = max(2, int(seq_len))
    vocab_size = max(5, int(vocab_size))
    n_classes = max(2, int(n_classes))

    x = np.random.randint(0, vocab_size, size=(n_samples, seq_len), dtype=np.int64)
    y = np.random.randint(0, n_classes, size=(n_samples,), dtype=np.int64)
    x_train, y_train, x_test, y_test = _shuffle_split(x, y, train_split)

    return {
        "dataset_id": str(uuid.uuid4()),
        "input_shape": [seq_len],
        "output_shape": [n_classes],
        "data_type": "text",
        "vocab_size": vocab_size,
        "train": {
            "x": x_train.tolist(),
            "y": _one_hot(y_train, n_classes).tolist(),
        },
        "test": {
            "x": x_test.tolist(),
            "y": _one_hot(y_test, n_classes).tolist(),
        },
    }

