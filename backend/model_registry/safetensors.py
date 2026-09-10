"""Minimal safetensors reader (NumPy only).

Reads the official Hugging Face ``.safetensors`` layout without requiring the
``safetensors`` package:

    8-byte little-endian header length -> UTF-8 JSON header -> raw tensors.

Each tensor is float32 here (the dacorvo/mnist-mlp checkpoint ships F32),
but the reader supports the common dtypes for robustness.
"""
from __future__ import annotations

import json
import struct

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": None,  # would need manual handling
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        (hdr_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
        data = fh.read()
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES[info["dtype"]]
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(data[start:end], dtype=dtype).reshape(info["shape"])
    return out
