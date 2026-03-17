from __future__ import annotations

import numpy as np


class GRULayer:
    layer_type = "gru"
    has_params = True

    def __init__(self, input_dim: int, hidden_dim: int, return_sequences: bool = False) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        concat_dim = input_dim + hidden_dim
        self.W_r = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.W_z = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.W_h = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.b_r = np.zeros(hidden_dim, dtype=np.float32)
        self.b_z = np.zeros(hidden_dim, dtype=np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)
        self.X: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.gates: dict | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, self.input_dim)
        self.X = x.astype(np.float32)
        T = self.X.shape[0]
        H = np.zeros((T, self.hidden_dim), dtype=np.float32)
        gates = {"r": [], "z": [], "h_tilde": []}
        h_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T):
            concat = np.concatenate([h_prev, self.X[t]])
            r = self._sigmoid(self.W_r @ concat + self.b_r)
            z = self._sigmoid(self.W_z @ concat + self.b_z)
            concat_h = np.concatenate([r * h_prev, self.X[t]])
            h_tilde = np.tanh(self.W_h @ concat_h + self.b_h)
            h = (1 - z) * h_prev + z * h_tilde
            gates["r"].append(r)
            gates["z"].append(z)
            gates["h_tilde"].append(h_tilde)
            H[t] = h
            h_prev = h
        self.H = H
        self.gates = {k: np.stack(v) for k, v in gates.items()}
        return H if self.return_sequences else H[-1]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X is None or self.H is None or self.gates is None:
            raise RuntimeError("GRULayer.backward called before forward.")
        T = self.X.shape[0]
        dW_r = np.zeros_like(self.W_r)
        dW_z = np.zeros_like(self.W_z)
        dW_h = np.zeros_like(self.W_h)
        db_r = np.zeros_like(self.b_r)
        db_z = np.zeros_like(self.b_z)
        db_h = np.zeros_like(self.b_h)
        dX = np.zeros_like(self.X)
        dh_next = np.zeros(self.hidden_dim, dtype=np.float32)
        if d_out.ndim == 1:
            d_out_seq = np.zeros_like(self.H)
            d_out_seq[-1] = d_out
        else:
            d_out_seq = d_out
        for t in reversed(range(T)):
            h_prev = self.H[t - 1] if t > 0 else np.zeros(self.hidden_dim, dtype=np.float32)
            r = self.gates["r"][t]
            z = self.gates["z"][t]
            h_tilde = self.gates["h_tilde"][t]
            dh = d_out_seq[t] + dh_next
            dh_tilde = dh * z
            dz = dh * (h_tilde - h_prev)
            dh_prev = dh * (1 - z)
            dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
            concat_h = np.concatenate([r * h_prev, self.X[t]])
            dW_h += np.outer(dh_tilde_raw, concat_h)
            db_h += dh_tilde_raw
            dconcat_h = self.W_h.T @ dh_tilde_raw
            dr = dconcat_h[: self.hidden_dim] * h_prev
            dx_h = dconcat_h[self.hidden_dim :]
            dr_raw = dr * r * (1 - r)
            concat = np.concatenate([h_prev, self.X[t]])
            dW_r += np.outer(dr_raw, concat)
            db_r += dr_raw
            dconcat_r = self.W_r.T @ dr_raw
            dz_raw = dz * z * (1 - z)
            dW_z += np.outer(dz_raw, concat)
            db_z += dz_raw
            dconcat_z = self.W_z.T @ dz_raw
            dh_next = dh_prev + dconcat_r[: self.hidden_dim] + dconcat_z[: self.hidden_dim] + dconcat_h[: self.hidden_dim] * r
            dX[t] = dx_h + dconcat_r[self.hidden_dim :] + dconcat_z[self.hidden_dim :]
        self.dW = np.concatenate([dW_r, dW_z, dW_h], axis=0)
        self.db = np.concatenate([db_r, db_z, db_h], axis=0)
        return dX

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        w = np.concatenate([self.W_r, self.W_z, self.W_h], axis=0)
        b = np.concatenate([self.b_r, self.b_z, self.b_h], axis=0)
        return w, b

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dW, self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        h = self.hidden_dim
        self.W_r = w[0:h].astype(np.float32)
        self.W_z = w[h : 2 * h].astype(np.float32)
        self.W_h = w[2 * h : 3 * h].astype(np.float32)
        self.b_r = b[0:h].astype(np.float32)
        self.b_z = b[h : 2 * h].astype(np.float32)
        self.b_h = b[2 * h : 3 * h].astype(np.float32)

