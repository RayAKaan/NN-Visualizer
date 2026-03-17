from __future__ import annotations

from typing import Tuple
import numpy as np


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, np.square(X[:, None, :] - X[None, :, :]).sum(axis=2)))


def _knn_graph(D: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    n = D.shape[0]
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    dist = np.take_along_axis(D, idx, axis=1)
    return idx, dist


def umap_project(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, n_iter: int = 400, min_dist: float = 0.1, learning_rate: float = 1.0) -> np.ndarray:
    X = X.astype(np.float32)
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, n_components), dtype=np.float32)

    D = _pairwise_dist(X)
    idx, dist = _knn_graph(D, min(n_neighbors, n - 1))

    # Compute fuzzy simplicial set weights
    sigmas = np.maximum(np.mean(dist, axis=1), 1e-6)
    weights = np.exp(-dist / sigmas[:, None])

    # Initialize with PCA-like projection
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_components]
    Y = (Xc @ eigvecs[:, order]).astype(np.float32)

    for _ in range(n_iter):
        for i in range(n):
            for n_i, j in enumerate(idx[i]):
                w = weights[i, n_i]
                diff = Y[i] - Y[j]
                dist_ij = np.sqrt(np.sum(diff ** 2)) + 1e-6
                # attractive force
                grad = w * diff / (dist_ij + min_dist)
                Y[i] -= learning_rate * grad
                Y[j] += learning_rate * grad

            # simple repulsion
            j = np.random.randint(0, n)
            if j != i:
                diff = Y[i] - Y[j]
                dist_ij = np.sqrt(np.sum(diff ** 2)) + 1e-6
                rep = diff / (dist_ij ** 2)
                Y[i] += learning_rate * 0.01 * rep

    Y = Y - Y.mean(axis=0)
    return Y.astype(np.float32)
