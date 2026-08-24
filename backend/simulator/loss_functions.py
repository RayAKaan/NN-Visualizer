from __future__ import annotations

from typing import Tuple

import numpy as np


def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    y_true = np.clip(y_true, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if not np.isfinite(loss).all():
        loss = np.nan_to_num(loss, nan=1.0, posinf=1.0, neginf=1.0)
    mean_loss = float(np.mean(loss))
    if not np.isfinite(mean_loss):
        mean_loss = 1.0
    eq = f"L = -[y·log(ŷ) + (1-y)·log(1-ŷ)] = {mean_loss:.3f}"
    return mean_loss, eq


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, str]:
    loss = 0.5 * (y_true - y_pred) ** 2
    if not np.isfinite(loss).all():
        loss = np.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
    mean_loss = float(np.mean(loss))
    if not np.isfinite(mean_loss):
        mean_loss = 0.0
    eq = f"L = 1/2 (y-ŷ)^2 = {mean_loss:.3f}"
    return mean_loss, eq


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, loss_fn: str) -> Tuple[float, str]:
    key = (loss_fn or "bce").lower()
    if key == "mse":
        return mse_loss(y_true, y_pred)
    return bce_loss(y_true, y_pred)

