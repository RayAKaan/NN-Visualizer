"""Weight/download helpers shared by the model adapters.

The registry never commits large binaries to git.  Instead each pretrained
weight file is downloaded lazily into a per-user cache directory (or the
project's ``models_store/`` directory, which is already git-ignored, when
``NN_MODEL_CACHE=local`` is set).

Cache precedence:
    1. ``$NN_MODEL_CACHE``     -> that directory
    2. ``$PROJECT_ROOT/models_store/pretrained`` when ``NN_MODEL_CACHE=local``
    3. ``~/.cache/nn_visualizer/models`` (default)
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request

from config import PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models_store")

_HF = "https://huggingface.co/{repo}/resolve/main/{filename}"


def cache_root() -> str:
    env = os.environ.get("NN_MODEL_CACHE", "").strip()
    if env.lower() == "local":
        root = os.path.join(MODELS_DIR, "pretrained")
    elif env:
        root = env
    else:
        root = os.path.join(os.path.expanduser("~"), ".cache", "nn_visualizer", "models")
    os.makedirs(root, exist_ok=True)
    return root


def hf_url(repo_id: str, filename: str) -> str:
    """Public (unauthenticated) download URL for a HF file."""
    return _HF.format(repo=repo_id, filename=filename)


def cache_path(*parts: str) -> str:
    return os.path.join(cache_root(), *parts)


def _download(url: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=os.path.dirname(dest) or ".", suffix=".part", delete=False).name
    req = urllib.request.Request(url, headers={"User-Agent": "nn-visualizer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return dest


def ensure_downloaded(url: str, filename: str, *, always: bool = False) -> str:
    """Download ``url`` into the model cache if not already present.

    Returns the local file path.  Files already cached (size > 0) are reused.
    """
    dest = cache_path(filename)
    if always and os.path.exists(dest):
        os.remove(dest)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    return _download(url, dest)


def write_provenance(model_id: str, info: dict) -> None:
    """Write a small sidecar JSON next to cached weights for reproducibility."""
    path = cache_path(f"{model_id.replace('/', '_')}.provenance.json")
    payload = {"model_id": model_id, "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), **info}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
