"""Small math / IO helpers used by adapters (numpy only where possible)."""
from __future__ import annotations

import base64
import io
import re

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    total = p.sum()
    return p / total if total > 0 else np.full_like(p, 1.0 / p.size)


def load_image_from_base64(image_b64: str, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Decode a base64 image (optionally a data URL) and resize to ``size``.

    Returns a float32 RGB array of shape ``size + (3,)`` with values in [0, 1].
    Requires Pillow.
    """
    from PIL import Image

    if "," in image_b64[:64] and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_image_from_path(path: str, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Decode an image file on disk and resize to ``size`` (values in [0,1])."""
    from PIL import Image

    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def decode_pixels(pixels: list[float] | None, expected: int = 784) -> np.ndarray:
    """Validate the legacy 784-element pixel array and return float32 [0,1]."""
    if not pixels:
        raise ValueError("'pixels' is required for image/MNIST models")
    arr = np.asarray(pixels, dtype=np.float32)
    if arr.size != expected:
        raise ValueError(f"Expected {expected} pixels, got {arr.size}")
    return arr.reshape(-1)


def mnist_words_split(text: str) -> list[str]:
    """Tokenizer pre-step for the keras-io IMDB BiLSTM.

    Mirrors ``keras.preprocessing.text.text_to_word_sequence`` defaults:
    lower-case the text, strip the keras default filter characters, and split
    on spaces (this is the pipeline the checkpoint was trained with).
    """
    FILTERS = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"
    for ch in FILTERS:
        text = text.replace(ch, " ")
    text = text.lower()
    return [w for w in text.split(" ") if w]
