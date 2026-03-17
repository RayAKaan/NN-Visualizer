from __future__ import annotations

from typing import List, Tuple
import numpy as np


def random_directions(param_vector: np.ndarray, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    d1 = rng.normal(size=param_vector.shape)
    d2 = rng.normal(size=param_vector.shape)
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 - d1 * np.dot(d1, d2)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    return d1.astype(np.float32), d2.astype(np.float32)


def param_vector_from_weights(weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    parts = []
    for w, b in zip(weights, biases):
        parts.append(w.reshape(-1))
        parts.append(b.reshape(-1))
    if not parts:
        return np.zeros(1, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def apply_vector_to_weights(
    base_weights: List[np.ndarray],
    base_biases: List[np.ndarray],
    direction1: np.ndarray,
    direction2: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    offset = direction1 * alpha + direction2 * beta
    idx = 0
    new_weights = []
    new_biases = []
    for w, b in zip(base_weights, base_biases):
        w_size = w.size
        b_size = b.size
        w_new = offset[idx : idx + w_size].reshape(w.shape) + w
        idx += w_size
        b_new = offset[idx : idx + b_size].reshape(b.shape) + b
        idx += b_size
        new_weights.append(w_new.astype(np.float32))
        new_biases.append(b_new.astype(np.float32))
    return new_weights, new_biases
