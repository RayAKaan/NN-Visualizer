from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

MODEL_PATH = Path(__file__).resolve().parent / "nn_visual.h5"


def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python train.py` to create it."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    # Force-build the graph (important for activation inspection)
    model(np.zeros((1, 784), dtype=np.float32))

    return model


def build_activation_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a STABLE activation model.

    This explicitly exposes ONLY Dense layers, in forward order:
    [hidden1 (128), hidden2 (64), output (10)]

    No InputLayer ambiguity. No indexing traps.
    """

    dense_outputs = [
        layer.output
        for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=dense_outputs,
        name="activation_model",
    )


def normalize_pixels(pixels: list[float]) -> np.ndarray:
    arr = np.array(pixels, dtype=np.float32)

    if arr.size != 784:
        raise ValueError("pixels must be length 784")

    return arr.reshape(1, 784)


def extract_weights(
    model: tf.keras.Model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns weight matrices:
    - input → hidden1  (784, 128)
    - hidden1 → hidden2 (128, 64)
    - hidden2 → output  (64, 10)
    """

    weights = model.get_weights()

    input_hidden = weights[0]
    hidden_hidden = weights[2]
    hidden_output = weights[4]

    return input_hidden, hidden_hidden, hidden_output
