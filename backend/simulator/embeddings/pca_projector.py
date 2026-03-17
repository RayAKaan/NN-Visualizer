from __future__ import annotations

from typing import Tuple
import numpy as np


def pca_project(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    comps = eigvecs[:, :n_components]
    projected = Xc @ comps
    variance = eigvals[:n_components] / max(np.sum(eigvals), 1e-12)
    return projected.astype(np.float32), variance.astype(np.float32)
