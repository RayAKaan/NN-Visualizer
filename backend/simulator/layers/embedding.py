from __future__ import annotations

import numpy as np


class EmbeddingLayer:
    layer_type = "embedding"
    has_params = True

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.E = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
        self.last_indices: np.ndarray | None = None
        self.dE: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        indices = x.astype(np.int64).reshape(-1)
        self.last_indices = indices
        return self.E[indices]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.last_indices is None:
            raise RuntimeError("EmbeddingLayer.backward called before forward.")
        self.dE = np.zeros_like(self.E)
        for i, idx in enumerate(self.last_indices):
            self.dE[idx] += d_out[i]
        return np.zeros_like(self.last_indices, dtype=np.float32)

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        return self.E, np.zeros((1,), dtype=np.float32)

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dE, np.zeros((1,), dtype=np.float32)

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        self.E = w.astype(np.float32)

