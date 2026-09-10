"""Unit tests for the small registry helpers (no network / no inference)."""
import base64
import io
import os

import numpy as np
import pytest

from model_registry import safetensors, store
from model_registry.util import (
    decode_pixels,
    load_image_from_base64,
    mnist_words_split,
    normalize_probs,
    softmax,
)


def test_softmax_normalizes():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert abs(p.sum() - 1) < 1e-9
    assert p[2] > p[1] > p[0]


def test_normalize_probs_handles_zero():
    p = normalize_probs(np.zeros(5))
    assert abs(p.sum() - 1) < 1e-9
    assert np.allclose(p, 0.2)


def test_decode_pixels_validates_length():
    with pytest.raises(ValueError):
        decode_pixels([0.0] * 10)
    arr = decode_pixels([0.5] * 784)
    assert arr.shape == (784,)


def test_mnist_words_split_lowercases_and_strips_punctuation():
    # mirrors keras text_to_word_sequence: apostrophes are NOT filter chars,
    # everything else in the keras default filter list is removed.
    words = mnist_words_split("  This film, (great)!  It's fine. ")
    assert words == ["this", "film", "great", "it's", "fine"]


def test_image_base64_roundtrip(tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (200, 100, 50)).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    arr = load_image_from_base64(b64, size=(224, 224))
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0

    # data-URL form also accepted
    arr2 = load_image_from_base64(f"data:image/png;base64,{b64}", size=(224, 224))
    assert arr2.shape == (224, 224, 3)


def test_safetensors_roundtrip(tmp_path):
    from PIL import Image

    p = tmp_path / "w.safetensors"
    tensors = {"a.weight": np.arange(12, dtype=np.float32).reshape(3, 4), "b.bias": np.ones(4, np.float32)}
    import json
    import struct

    with open(p, "wb") as fh:
        header = {}
        offset = 0
        body = b""
        for name, arr in tensors.items():
            raw = np.ascontiguousarray(arr).tobytes()
            header[name] = {"dtype": "F32", "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
            offset += len(raw)
            body += raw
        hdr = json.dumps(header, separators=(",", ":")).encode()
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        fh.write(body)
    out = safetensors.load_safetensors(str(p))
    assert set(out) == {"a.weight", "b.bias"}
    assert np.allclose(out["a.weight"], tensors["a.weight"])


def test_cache_root_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NN_MODEL_CACHE", str(tmp_path / "cache"))
    assert store.cache_root() == str(tmp_path / "cache")
    monkeypatch.setenv("NN_MODEL_CACHE", "local")
    root = store.cache_root()
    assert root.endswith(os.path.join("models_store", "pretrained"))
