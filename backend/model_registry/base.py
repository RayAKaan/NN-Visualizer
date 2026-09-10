"""Adapter interface implemented by every registry model."""
from __future__ import annotations

import abc
import threading
from typing import Any


class ModelLoadError(RuntimeError):
    """Raised when a pretrained checkpoint cannot be loaded/used."""


class ModelAdapter(abc.ABC):
    """Contract for every Prediction-Mode model.

    Each adapter is responsible for:

    * declaring its metadata (``metadata`` property)
    * loading its weights lazily (``load``) and caching them
    * accepting a family-appropriate raw input and producing a
      standardized ``(predicted_index, confidence, probabilities)`` tuple
    * optionally exposing an ``extra`` visualization payload
    """

    # unique registry id
    model_id: str = ""

    def __init__(self):
        self._loaded = False
        self._load_lock = threading.Lock()
        self._load_error: str | None = None
        self.metadata = self._describe()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def _describe(self) -> dict[str, Any]:
        """Return the metadata dict for this model (see schema.MODEL_META_FIELDS)."""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self) -> Any:
        """Load (and cache) the underlying model. Must be idempotent."""

    def ensure_loaded(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            try:
                self.load()
                self._loaded = True
                self._load_error = None
                self.refresh_metadata()
            except Exception as exc:  # noqa: BLE001 - surfaced as status
                self._loaded = False
                self._load_error = str(exc)
                raise

    def unload(self) -> None:
        with self._load_lock:
            self._release()
            self._loaded = False
            self._load_error = None

    def _release(self) -> None:
        """Drop cached weights / models (default: no-op)."""

    def check_available(self) -> tuple[bool, str | None]:
        """Whether the model can be used right now.

        Returns ``(ok, reason)``.  Adapters whose weights live on disk override
        this to probe for the file; network-downloaded models default to True
        (a download may still fail at load time, which is reported as an error).
        """
        return True, None

    def refresh_metadata(self) -> None:
        """Opportunistically enrich metadata (e.g. parameter_count) after load."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def predict(self, raw_input: dict[str, Any]) -> tuple[int, float, list[float], list[str] | None, dict | None]:
        """Run one inference.

        Args:
            raw_input: family-appropriate payload, e.g.
                ``{"pixels": [784 floats]}``, ``{"image": b64 str}`` or
                ``{"text": "..."}``.

        Returns:
            (predicted_index, confidence, probabilities, labels, extra)
            where probabilities are aligned with ``labels``.
        """
