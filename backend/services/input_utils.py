"""Shared input preparation and dataset helpers for Lab endpoints.

These live outside ``api.lab`` so the execution-trace service can reuse the exact
same normalization without importing the router (which would create an import
cycle).
"""
from typing import Any

import numpy as np
import tensorflow as tf
from fastapi import HTTPException


def arch_key(architecture: str) -> str:
    key = architecture.strip().lower()
    if key in ("ann", "cnn", "rnn"):
        return key
    raise HTTPException(status_code=400, detail=f"Unsupported architecture: {architecture}")


def prepare_input(pixels: list[float], arch: str, dataset: str) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)
    if dataset == "catdog" and arr.size >= 3 * 64 * 64:
        rgb = arr[: 3 * 64 * 64].reshape(3, 64, 64)
        gray = np.mean(rgb, axis=0)
        arr = tf.image.resize(gray[..., np.newaxis], [28, 28]).numpy().reshape(-1)
    if arr.size < 28 * 28:
        padded = np.zeros(28 * 28, dtype=np.float32)
        padded[: arr.size] = arr
        arr = padded
    elif arr.size > 28 * 28:
        arr = arr[: 28 * 28]
    if arch == "ann":
        return arr.reshape(1, 784)
    if arch == "cnn":
        return arr.reshape(1, 28, 28, 1)
    return arr.reshape(1, 28, 28)


def dataset_adjust(output_data: np.ndarray, dataset: str) -> np.ndarray:
    if dataset != "catdog":
        return output_data
    if output_data.size < 2:
        return np.pad(output_data, (0, 2 - output_data.size))
    two = output_data[:2].astype(np.float32)
    denom = float(np.sum(two))
    return two / denom if denom > 0 else two


def binary_probs(probs: tf.Tensor, dataset: str) -> tf.Tensor:
    if dataset != "catdog":
        return probs
    two = probs[:, :2]
    return two / (tf.reduce_sum(two, axis=1, keepdims=True) + 1e-7)


def label_for_dataset(dataset: str, label: int) -> int | str:
    if dataset == "catdog":
        return "Cat" if label == 0 else "Dog"
    return int(label)


def label_text_for_dataset(dataset: str, label: int) -> str:
    if dataset == "catdog":
        return "Cat" if label == 0 else "Dog"
    return str(int(label))


def tensor_stats(values: Any) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "absMax": 0.0,
                "sparsity": 0.0, "positive": 0.0, "negative": 0.0, "zero": 0.0}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "absMax": float(np.max(np.abs(x))),
        "sparsity": float(np.mean(np.abs(x) < 1e-4)),
        "positive": float(np.mean(x > 0.0)),
        "negative": float(np.mean(x < 0.0)),
        "zero": float(np.mean(np.abs(x) <= 1e-7)),
    }