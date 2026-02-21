from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from config import MODEL_PATH


def load_model(path: str | None = None) -> tf.keras.Model:
    model_path = MODEL_PATH if path is None else path
    if not tf.io.gfile.exists(str(model_path)):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python train.py` to create it."
        )
    model = tf.keras.models.load_model(model_path)
    model.predict(np.zeros((1, 784), dtype=np.float32), verbose=0)
    return model


def build_activation_model(model: tf.keras.Model) -> tf.keras.Model:
    if not model.inputs:
        model.predict(np.zeros((1, 784), dtype=np.float32), verbose=0)
    return tf.keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])


def normalize_pixels(pixels: list[float]) -> np.ndarray:
    arr = np.array(pixels, dtype=np.float32)
    if arr.size != 784:
        raise ValueError("pixels must be length 784")
    return arr.reshape(1, 784)


def extract_weights(model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = model.get_weights()
    input_hidden = weights[0]
    hidden_hidden = weights[2]
    hidden_output = weights[4]
    return input_hidden, hidden_hidden, hidden_output
