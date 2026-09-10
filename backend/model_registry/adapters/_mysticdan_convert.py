"""Standalone pure-Python converter for the mysticdan JAX checkpoint.

The JAX checkpoint stores plain NumPy payloads but wraps them in
``jax.Array`` objects whose pickle opcodes reference ``jax._src.array``.
This module registers a light shim that reconstructs exactly the underlying
``numpy.ndarray`` (mirroring ``jax._src.array._reconstruct_array`` but
skipping ``device_put``), so the file can be converted on a machine that does
not have JAX installed.  Run as:

    python -m model_registry.adapters._mysticdan_convert <input.pkl> <output.npz>

Only NumPy is imported here — deliberately no TensorFlow/Keras/JAX.
"""
from __future__ import annotations

import os
import pickle
import sys
import types

import numpy as np


def _install_shim() -> None:
    jax = types.ModuleType("jax")
    jax_src = types.ModuleType("jax._src")
    jax_array = types.ModuleType("jax._src.array")

    def _reconstruct_array(fun, args, arr_state, aval_state):
        np_value = fun(*args)
        np_value.__setstate__(arr_state)
        return np_value

    jax_array._reconstruct_array = _reconstruct_array
    jax._src = jax_src
    jax_src.array = jax_array
    sys.modules["jax"] = jax
    sys.modules["jax._src"] = jax_src
    sys.modules["jax._src.array"] = jax_array


def convert(pkl_path: str, npz_path: str) -> None:
    _install_shim()
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    params = data["params"]
    out: dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(params):
        out[f"w{i}"] = np.asarray(w, dtype=np.float32)
        out[f"b{i}"] = np.asarray(b, dtype=np.float32)
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    np.savez(npz_path, **out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python -m model_registry.adapters._mysticdan_convert <in.pkl> <out.npz>")
        raise SystemExit(2)
    convert(sys.argv[1], sys.argv[2])
