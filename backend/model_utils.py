from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# ============================================================
# Config
# ============================================================

MODEL_PATH = Path(__file__).resolve().parent / "nn_visual.h5"

# ============================================================
# Model loading
# ============================================================

def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # 🔴 Force graph build
    model(np.zeros((1, model.input_shape[-1]), dtype=np.float32))

    return model


# ============================================================
# Activation model
# ============================================================

def build_activation_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Returns outputs of ALL Dense layers in forward order.
    Last Dense is always the softmax output.
    """

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    if len(dense_layers) < 2:
        raise ValueError("Model must contain at least one hidden Dense layer.")

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in dense_layers],
        name="activation_model",
    )


# ============================================================
# Input normalization
# ============================================================

def normalize_pixels(pixels: List[float]) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)

    if arr.ndim != 1:
        raise ValueError("pixels must be a flat list")

    return arr.reshape(1, -1)


# ============================================================
# Weight extraction (SAFE & DYNAMIC)
# ============================================================

def extract_weights(
    model: tf.keras.Model,
) -> List[np.ndarray]:
    """
    Returns Dense kernel matrices ONLY, in forward order.

    Example:
    [
        (784, 128),
        (128, 64),
        (64, 10)
    ]
    """

    kernels: List[np.ndarray] = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            kernel, _bias = layer.get_weights()
            kernels.append(kernel)

    if len(kernels) < 2:
        raise ValueError("Expected at least 2 Dense layers.")

    return kernels