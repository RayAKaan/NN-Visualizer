from __future__ import annotations

import math
import uuid
from typing import Dict, List, Tuple

import numpy as np


def _shuffle_split(x: np.ndarray, y: np.ndarray, train_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    cut = int(len(x) * train_split)
    return x[:cut], y[:cut], x[cut:], y[cut:]


def _to_points(x: np.ndarray, y: np.ndarray) -> List[dict]:
    out = []
    for i in range(len(x)):
        out.append({"x": x[i].tolist(), "y": [int(y[i])]})
    return out


def _stats(x: np.ndarray, y: np.ndarray) -> Dict:
    return {
        "n_train": int(len(x)),
        "class_balance": {str(c): int(np.sum(y == c)) for c in np.unique(y)},
        "feature_range": {
            "x1": [float(np.min(x[:, 0])), float(np.max(x[:, 0]))],
            "x2": [float(np.min(x[:, 1])), float(np.max(x[:, 1]))],
        },
        "feature_mean": [float(np.mean(x[:, 0])), float(np.mean(x[:, 1]))],
        "feature_std": [float(np.std(x[:, 0])), float(np.std(x[:, 1]))],
    }


def _make_circle(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.random.rand(n) * 2 * math.pi
    radii = np.random.rand(n)
    x = np.vstack([np.cos(angles) * radii, np.sin(angles) * radii]).T
    y = (radii > 0.5).astype(int)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def _make_xor(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1, 1, size=(n, 2))
    y = (x[:, 0] * x[:, 1] < 0).astype(int)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def _make_spiral(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    n_per = n // 2
    theta = np.sqrt(np.random.rand(n_per)) * 2 * math.pi
    r = 2 * theta
    x1 = np.vstack([r * np.cos(theta), r * np.sin(theta)]).T
    x2 = np.vstack([r * np.cos(theta + math.pi), r * np.sin(theta + math.pi)]).T
    x = np.vstack([x1, x2]) / (2 * math.pi)
    y = np.array([0] * n_per + [1] * n_per)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def _make_moons(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    n_per = n // 2
    theta = np.linspace(0, math.pi, n_per)
    x1 = np.vstack([np.cos(theta), np.sin(theta)]).T
    x2 = np.vstack([1 - np.cos(theta), 1 - np.sin(theta) - 0.5]).T
    x = np.vstack([x1, x2])
    y = np.array([0] * n_per + [1] * n_per)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def _make_blobs(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    n_per = n // 2
    c1 = np.random.normal(loc=[-1, -1], scale=noise + 0.3, size=(n_per, 2))
    c2 = np.random.normal(loc=[1, 1], scale=noise + 0.3, size=(n_per, 2))
    x = np.vstack([c1, c2])
    y = np.array([0] * n_per + [1] * n_per)
    return x, y


def _make_linear(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1, 1, size=(n, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def _make_rings(n: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.random.rand(n) * 2 * math.pi
    radii = np.where(np.random.rand(n) > 0.5, 1.0, 0.5)
    x = np.vstack([np.cos(angles) * radii, np.sin(angles) * radii]).T
    y = (radii > 0.75).astype(int)
    x += np.random.normal(0.0, noise, x.shape)
    return x, y


def generate_dataset(dataset_type: str, n_samples: int, noise: float, train_split: float, seed: int | None = None) -> Dict:
    if seed is not None:
        np.random.seed(seed)
    n_samples = max(10, int(n_samples))
    noise = float(noise)
    train_split = float(train_split)

    dtype = (dataset_type or "circle").lower()
    if dtype == "circle":
        x, y = _make_circle(n_samples, noise)
    elif dtype == "xor":
        x, y = _make_xor(n_samples, noise)
    elif dtype == "spiral":
        x, y = _make_spiral(n_samples, noise)
    elif dtype == "moons":
        x, y = _make_moons(n_samples, noise)
    elif dtype == "blobs":
        x, y = _make_blobs(n_samples, noise)
    elif dtype == "linear":
        x, y = _make_linear(n_samples, noise)
    elif dtype == "rings":
        x, y = _make_rings(n_samples, noise)
    else:
        raise ValueError("Unknown dataset type.")

    x_train, y_train, x_test, y_test = _shuffle_split(x, y, train_split)
    return {
        "dataset_id": str(uuid.uuid4()),
        "train": _to_points(x_train, y_train),
        "test": _to_points(x_test, y_test),
        "stats": {
            **_stats(x_train, y_train),
            "n_test": int(len(x_test)),
        },
    }


def custom_dataset(points: List[dict], train_split: float) -> Dict:
    if not points:
        raise ValueError("No points provided.")
    x = np.array([p["x"] for p in points], dtype=np.float32)
    y = np.array([p["y"][0] for p in points], dtype=np.int64)
    x_train, y_train, x_test, y_test = _shuffle_split(x, y, train_split)
    return {
        "dataset_id": str(uuid.uuid4()),
        "train": _to_points(x_train, y_train),
        "test": _to_points(x_test, y_test),
        "stats": {
            **_stats(x_train, y_train),
            "n_test": int(len(x_test)),
        },
    }

