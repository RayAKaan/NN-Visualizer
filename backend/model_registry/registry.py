"""Central Model Registry for Prediction Mode.

The registry holds every model (metadata + lazy-loading adapter), tracks load
state and exposes model-agnostic operations used by the Prediction API.
"""
from __future__ import annotations

import threading
from typing import Any

from model_registry.base import ModelAdapter
from model_registry.schema import (
    FAMILIES,
    STATUS_AVAILABLE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_UNAVAILABLE,
)

# ---------------------------------------------------------------------------
# Adapter registry (imported here to avoid circular imports).
# ---------------------------------------------------------------------------
from model_registry.adapters.ann_adapters import (
    DacorvoMnistMlpAdapter,
    MysticdanMnistMlpAdapter,
    NNVisualizerAnnAdapter,
)
from model_registry.adapters.cnn_adapters import (
    MobileNetV2Adapter,
    MobileNetV3SmallAdapter,
    ResNet50Adapter,
    VGG16Adapter,
)
from model_registry.adapters.rnn_imdb_kerasio import KerasIoImdbBiLSTMAdapter
from model_registry.adapters.legacy_mnist_adapters import LegacyCNNAdapter, LegacyRNNAdapter

# Every runnable model.  Ordering controls UI sort order.
ADAPTER_CLASSES: list[type[ModelAdapter]] = [
    DacorvoMnistMlpAdapter,
    MysticdanMnistMlpAdapter,
    NNVisualizerAnnAdapter,
    MobileNetV3SmallAdapter,
    MobileNetV2Adapter,
    ResNet50Adapter,
    VGG16Adapter,
    KerasIoImdbBiLSTMAdapter,
    LegacyCNNAdapter,
    LegacyRNNAdapter,
]

# Intentionally-listed models that were *verified* to be incompatible with the
# current CPU / TensorFlow stack (kept so the UI / docs can explain why they
# are not integrated).  status = "unavailable".
UNAVAILABLE_MODELS: list[dict[str, Any]] = [
    {
        "id": "rnn-jongador-lstm-imdb-256",
        "name": "jongador LSTM IMDB (256)",
        "family": "RNN",
        "framework": "PyTorch / TextAttack",
        "source": "Hugging Face — jongador/lstm-imdb-256",
        "pretrained": True,
        "unavailable_reason": (
            "Deprecated TextAttack checkpoint (~320 MB, embeds GloVe-200d). Requires the "
            "legacy textattack + torch stack and would bloat CPU startup/memory."
        ),
    },
    {
        "id": "rnn-jongador-lstm-imdb-512",
        "name": "jongador LSTM IMDB (512)",
        "family": "RNN",
        "framework": "PyTorch / TextAttack",
        "source": "Hugging Face — jongador/lstm-imdb-512",
        "pretrained": True,
        "unavailable_reason": (
            "TextAttack-format checkpoint (torch/textattack + GloVe loading); not loadable "
            "in the project's TensorFlow/Keras dependency stack."
        ),
    },
    {
        "id": "rnn-pratyushee-assamese-lstm",
        "name": "Assamese LSTM sentiment",
        "family": "RNN",
        "framework": "TensorFlow (notebook only)",
        "source": "Hugging Face — pratyushee/assamese-sentiment-analysis",
        "pretrained": False,
        "unavailable_reason": (
            "Repository contains only a training notebook/requirements — no pretrained "
            "checkpoint file is published, so nothing can be loaded."
        ),
    },
]


class ModelRegistry:
    """Holds metadata + lazy adapters for every registered model."""

    def __init__(self, adapter_classes: list[type[ModelAdapter]] | None = None):
        self._lock = threading.RLock()
        self._adapters: dict[str, ModelAdapter] = {}
        classes = list(adapter_classes) if adapter_classes is not None else list(ADAPTER_CLASSES)
        for cls in classes:
            adapter = cls()
            self._adapters[adapter.model_id] = adapter

    # ------------------------------------------------------------------
    def ids(self, family: str | None = None) -> list[str]:
        with self._lock:
            ids = [a.model_id for a in self._adapters.values()]
        if family:
            fam = family.upper()
            ids = [i for i in ids if self.get(i).metadata["family"] == fam]
        return ids

    def families(self) -> list[str]:
        with self._lock:
            return [fam for fam in FAMILIES if any(a.metadata["family"] == fam for a in self._adapters.values())]

    def get(self, model_id: str) -> ModelAdapter:
        with self._lock:
            adapter = self._adapters.get(model_id)
        if adapter is None:
            raise KeyError(model_id)
        return adapter

    def contains(self, model_id: str) -> bool:
        return model_id in self._adapters

    def unavailable_reason(self, model_id: str) -> str | None:
        """Return the reason string when ``model_id`` is an intentionally
        listed-but-unavailable model, else None."""
        for m in UNAVAILABLE_MODELS:
            if m["id"] == model_id:
                return m.get("unavailable_reason")
        return None

    # ------------------------------------------------------------------
    def load(self, model_id: str) -> ModelAdapter:
        adapter = self.get(model_id)
        adapter.ensure_loaded()
        return adapter

    def unload(self, model_id: str) -> None:
        adapter = self.get(model_id)
        adapter.unload()

    def loaded_ids(self) -> list[str]:
        return [aid for aid, a in self._adapters.items() if a.is_loaded]

    def status(self, model_id: str) -> dict[str, Any]:
        adapter = self.get(model_id)
        meta = dict(adapter.metadata)
        if adapter.is_loaded:
            meta["status"] = STATUS_LOADED
            return meta
        if adapter.load_error:
            meta["status"] = STATUS_ERROR
            meta["error"] = adapter.load_error
            return meta
        ok, reason = adapter.check_available()
        if not ok:
            meta["status"] = STATUS_UNAVAILABLE
            meta["unavailable_reason"] = reason
            return meta
        meta["status"] = STATUS_AVAILABLE
        return meta

    # ------------------------------------------------------------------
    def catalog(self) -> list[dict[str, Any]]:
        """Metadata + runtime status for every model, grouped sensibly."""
        with self._lock:
            rows = [self.status(aid) for aid in self._adapters]
            rows.sort(key=lambda r: (r["family"], r["name"]))
            unavailable = [
                {**m, "status": STATUS_UNAVAILABLE}
                for m in UNAVAILABLE_MODELS
            ]
            return rows + unavailable


registry = ModelRegistry()
