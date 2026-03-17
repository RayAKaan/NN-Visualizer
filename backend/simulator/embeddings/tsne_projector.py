from __future__ import annotations

from typing import Tuple
import numpy as np


def _hbeta(dist: np.ndarray, beta: float) -> Tuple[float, np.ndarray]:
    p = np.exp(-dist * beta)
    sum_p = np.sum(p)
    if sum_p == 0:
        return 0.0, np.zeros_like(p)
    h = np.log(sum_p) + beta * np.sum(dist * p) / sum_p
    return h, p / sum_p


def _x2p(X: np.ndarray, perplexity: float, tol: float = 1e-5) -> np.ndarray:
    n = X.shape[0]
    D = np.square(X[:, None, :] - X[None, :, :]).sum(axis=2)
    P = np.zeros((n, n), dtype=np.float32)
    beta = np.ones((n, 1), dtype=np.float32)
    logU = np.log(perplexity)

    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = np.concatenate((D[i, :i], D[i, i+1:]))
        h, thisP = _hbeta(Di, beta[i])
        hdiff = h - logU
        tries = 0
        while np.abs(hdiff) > tol and tries < 50:
            if hdiff > 0:
                betamin = beta[i].copy()
                beta[i] = beta[i] * 2 if np.isinf(betamax) else (beta[i] + betamax) / 2
            else:
                betamax = beta[i].copy()
                beta[i] = beta[i] / 2 if np.isinf(betamin) else (beta[i] + betamin) / 2
            h, thisP = _hbeta(Di, beta[i])
            hdiff = h - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P


def tsne_project(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0, n_iter: int = 500, learning_rate: float = 200.0) -> np.ndarray:
    X = X.astype(np.float32)
    n = X.shape[0]
    if n <= 1:
        return np.zeros((n, n_components), dtype=np.float32)

    P = _x2p(X, perplexity)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    Y = np.random.normal(0.0, 1e-4, size=(n, n_components)).astype(np.float32)
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    momentum = 0.5

    for it in range(n_iter):
        sum_Y = np.sum(np.square(Y), axis=1)
        num = -2.0 * np.dot(Y, Y.T)
        num = 1.0 / (1.0 + np.add(np.add(num, sum_Y).T, sum_Y))
        np.fill_diagonal(num, 0.0)
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        for i in range(n):
            dY[i, :] = 4.0 * np.sum((PQ[:, i] * num[:, i])[:, None] * (Y[i, :] - Y), axis=0)

        momentum = 0.8 if it > 100 else 0.5
        iY = momentum * iY - learning_rate * dY
        Y = Y + iY
        Y = Y - np.mean(Y, axis=0)

    return Y.astype(np.float32)
