from __future__ import annotations

import numpy as np


class LSTMLayer:
    layer_type = "lstm"
    has_params = True

    def __init__(self, input_dim: int, hidden_dim: int, return_sequences: bool = False) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        concat_dim = input_dim + hidden_dim
        self.W_f = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.W_i = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.W_o = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.W_c = np.random.randn(hidden_dim, concat_dim).astype(np.float32) * 0.1
        self.b_f = np.zeros(hidden_dim, dtype=np.float32)
        self.b_i = np.zeros(hidden_dim, dtype=np.float32)
        self.b_o = np.zeros(hidden_dim, dtype=np.float32)
        self.b_c = np.zeros(hidden_dim, dtype=np.float32)
        self.X: np.ndarray | None = None
        self.H: np.ndarray | None = None
        self.C: np.ndarray | None = None
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
        C = np.zeros((T, self.hidden_dim), dtype=np.float32)
        gates = {"f": [], "i": [], "o": [], "g": []}
        h_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        c_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T):
            concat = np.concatenate([h_prev, self.X[t]])
            f = self._sigmoid(self.W_f @ concat + self.b_f)
            i = self._sigmoid(self.W_i @ concat + self.b_i)
            o = self._sigmoid(self.W_o @ concat + self.b_o)
            g = np.tanh(self.W_c @ concat + self.b_c)
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            gates["f"].append(f)
            gates["i"].append(i)
            gates["o"].append(o)
            gates["g"].append(g)
            H[t] = h
            C[t] = c
            h_prev = h
            c_prev = c
        self.H = H
        self.C = C
        self.gates = {k: np.stack(v) for k, v in gates.items()}
        return H if self.return_sequences else H[-1]

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self.X is None or self.H is None or self.C is None or self.gates is None:
            raise RuntimeError("LSTMLayer.backward called before forward.")
        T = self.X.shape[0]
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_o = np.zeros_like(self.W_o)
        dW_c = np.zeros_like(self.W_c)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_o = np.zeros_like(self.b_o)
        db_c = np.zeros_like(self.b_c)
        dX = np.zeros_like(self.X)
        dh_next = np.zeros(self.hidden_dim, dtype=np.float32)
        dc_next = np.zeros(self.hidden_dim, dtype=np.float32)
        if d_out.ndim == 1:
            d_out_seq = np.zeros_like(self.H)
            d_out_seq[-1] = d_out
        else:
            d_out_seq = d_out

        for t in reversed(range(T)):
            h_prev = self.H[t - 1] if t > 0 else np.zeros(self.hidden_dim, dtype=np.float32)
            c_prev = self.C[t - 1] if t > 0 else np.zeros(self.hidden_dim, dtype=np.float32)
            f = self.gates["f"][t]
            i = self.gates["i"][t]
            o = self.gates["o"][t]
            g = self.gates["g"][t]
            c = self.C[t]
            tanh_c = np.tanh(c)
            dh = d_out_seq[t] + dh_next
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            do_raw = do * o * (1 - o)
            dg_raw = dg * (1 - g ** 2)

            concat = np.concatenate([h_prev, self.X[t]])
            dW_f += np.outer(df_raw, concat)
            dW_i += np.outer(di_raw, concat)
            dW_o += np.outer(do_raw, concat)
            dW_c += np.outer(dg_raw, concat)
            db_f += df_raw
            db_i += di_raw
            db_o += do_raw
            db_c += dg_raw

            dconcat = (
                self.W_f.T @ df_raw
                + self.W_i.T @ di_raw
                + self.W_o.T @ do_raw
                + self.W_c.T @ dg_raw
            )
            dh_next = dconcat[: self.hidden_dim]
            dc_next = dc * f
            dX[t] = dconcat[self.hidden_dim :]

        self.dW = np.concatenate([dW_f, dW_i, dW_o, dW_c], axis=0)
        self.db = np.concatenate([db_f, db_i, db_o, db_c], axis=0)
        return dX

    def params(self) -> tuple[np.ndarray, np.ndarray]:
        w = np.concatenate([self.W_f, self.W_i, self.W_o, self.W_c], axis=0)
        b = np.concatenate([self.b_f, self.b_i, self.b_o, self.b_c], axis=0)
        return w, b

    def grads(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.dW, self.db

    def set_params(self, w: np.ndarray, b: np.ndarray) -> None:
        h = self.hidden_dim
        self.W_f = w[0:h].astype(np.float32)
        self.W_i = w[h : 2 * h].astype(np.float32)
        self.W_o = w[2 * h : 3 * h].astype(np.float32)
        self.W_c = w[3 * h : 4 * h].astype(np.float32)
        self.b_f = b[0:h].astype(np.float32)
        self.b_i = b[h : 2 * h].astype(np.float32)
        self.b_o = b[2 * h : 3 * h].astype(np.float32)
        self.b_c = b[3 * h : 4 * h].astype(np.float32)

