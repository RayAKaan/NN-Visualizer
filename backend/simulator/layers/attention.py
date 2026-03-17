from __future__ import annotations

import numpy as np


class AttentionLayer:
    layer_type = "attention"
    has_params = True

    def __init__(self, d_model: int, num_heads: int = 1) -> None:
        self.d_model = d_model
        self.num_heads = max(1, num_heads)
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.W_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
        self.last_input: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.O: np.ndarray | None = None
        self.attn_weights: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, self.d_model)
        self.last_input = x.astype(np.float32)
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        scale = np.sqrt(self.d_model)
        scores = (Q @ K.T) / scale
        exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        A = exp / np.sum(exp, axis=-1, keepdims=True)
        self.Q = Q
        self.K = K
        self.V = V
        self.A = A
        self.attn_weights = A
        O = A @ V
        self.O = O
        return O @ self.W_o

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.last_input is None or self.Q is None or self.K is None or self.V is None or self.A is None or self.O is None:
            raise RuntimeError("AttentionLayer.backward called before forward.")
        X = self.last_input
        scale = np.sqrt(self.d_model)
        dY = d_out
        dW_o = self.O.T @ dY
        dO = dY @ self.W_o.T
        dA = dO @ self.V.T
        dV = self.A.T @ dO

        dS = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            ai = self.A[i]
            dai = dA[i]
            dS[i] = ai * (dai - np.sum(dai * ai))

        dQ = dS @ self.K / scale
        dK = dS.T @ self.Q / scale

        dW_q = X.T @ dQ
        dW_k = X.T @ dK
        dW_v = X.T @ dV

        dX = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T

        self.dW = np.stack([dW_q, dW_k, dW_v, dW_o], axis=0)
        self.db = np.zeros((1,), dtype=np.float32)
        return dX

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        # pack all weights in a single matrix for the optimizer
        w = np.stack([self.W_q, self.W_k, self.W_v, self.W_o], axis=0)
        return w, np.zeros((1,), dtype=np.float32)

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dW, self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        if w.ndim == 4 and w.shape[0] >= 4:
            self.W_q = w[0].astype(np.float32)
            self.W_k = w[1].astype(np.float32)
            self.W_v = w[2].astype(np.float32)
            self.W_o = w[3].astype(np.float32)
