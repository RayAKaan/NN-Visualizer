from __future__ import annotations

import numpy as np


class RNNLayer:
    layer_type = "rnn"
    has_params = True

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh", return_sequences: bool = False) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.return_sequences = return_sequences
        self.W_xh = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.1
        self.W_hh = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.1
        self.b = np.zeros(hidden_dim, dtype=np.float32)
        self.X: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.dW_xh: np.ndarray | None = None
        self.dW_hh: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def _act(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _act_deriv(self, x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1.0 - t * t

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, self.input_dim)
        self.X = x.astype(np.float32)
        T = self.X.shape[0]
        H = np.zeros((T, self.hidden_dim), dtype=np.float32)
        Z = np.zeros_like(H)
        h_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T):
            z = self.W_xh @ self.X[t] + self.W_hh @ h_prev + self.b
            h = self._act(z)
            Z[t] = z
            H[t] = h
            h_prev = h
        self.H = H
        self.Z = Z
        return H if self.return_sequences else H[-1]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X is None or self.H is None or self.Z is None:
            raise RuntimeError("RNNLayer.backward called before forward.")
        T = self.X.shape[0]
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.X)
        dh_next = np.zeros(self.hidden_dim, dtype=np.float32)
        if d_out.ndim == 1:
            d_out_seq = np.zeros_like(self.H)
            d_out_seq[-1] = d_out
        else:
            d_out_seq = d_out
        for t in reversed(range(T)):
            dh = d_out_seq[t] + dh_next
            dz = dh * self._act_deriv(self.Z[t])
            db += dz
            dW_xh += np.outer(dz, self.X[t])
            h_prev = self.H[t - 1] if t > 0 else np.zeros(self.hidden_dim, dtype=np.float32)
            dW_hh += np.outer(dz, h_prev)
            dX[t] = self.W_xh.T @ dz
            dh_next = self.W_hh.T @ dz
        self.dW_xh = dW_xh
        self.dW_hh = dW_hh
        self.db = db
        return dX

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        # pack weights as a single matrix for optimizer convenience
        w = np.concatenate([self.W_xh, self.W_hh], axis=1)
        return w, self.b

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.dW_xh is None or self.dW_hh is None:
            return None, None
        return np.concatenate([self.dW_xh, self.dW_hh], axis=1), self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        split = self.input_dim
        self.W_xh = w[:, :split].astype(np.float32)
        self.W_hh = w[:, split:].astype(np.float32)
        self.b = b.astype(np.float32)

