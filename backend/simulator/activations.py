from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ActivationSpec:
    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray], np.ndarray]
    formula: str
    latex: str


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    t = np.tanh(x)
    return 1.0 - t * t


def _leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _leaky_relu_deriv(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, 1.0, alpha).astype(np.float32)


def _linear(x: np.ndarray) -> np.ndarray:
    return x


def _linear_deriv(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


_ACTIVATIONS: Dict[str, ActivationSpec] = {
    "relu": ActivationSpec(
        name="relu",
        forward=_relu,
        derivative=_relu_deriv,
        formula="f(z) = max(0, z)",
        latex=r"f(z) = \max(0, z)",
    ),
    "sigmoid": ActivationSpec(
        name="sigmoid",
        forward=_sigmoid,
        derivative=_sigmoid_deriv,
        formula="f(z) = 1/(1+e^(-z))",
        latex=r"f(z) = \frac{1}{1 + e^{-z}}",
    ),
    "tanh": ActivationSpec(
        name="tanh",
        forward=_tanh,
        derivative=_tanh_deriv,
        formula="f(z) = tanh(z)",
        latex=r"f(z) = \tanh(z)",
    ),
    "leaky_relu": ActivationSpec(
        name="leaky_relu",
        forward=_leaky_relu,
        derivative=_leaky_relu_deriv,
        formula="f(z) = z if z>0 else alpha*z",
        latex=r"f(z) = \begin{cases} z & z>0 \\ \alpha z & z \le 0 \end{cases}",
    ),
    "linear": ActivationSpec(
        name="linear",
        forward=_linear,
        derivative=_linear_deriv,
        formula="f(z) = z",
        latex=r"f(z) = z",
    ),
}


def get_activation(name: str) -> ActivationSpec:
    key = (name or "linear").lower()
    if key not in _ACTIVATIONS:
        return _ACTIVATIONS["linear"]
    return _ACTIVATIONS[key]


def activation_plot_points(name: str, domain: Tuple[float, float] = (-3.0, 3.0), steps: int = 13) -> list[list[float]]:
    spec = get_activation(name)
    xs = np.linspace(domain[0], domain[1], steps)
    ys = spec.forward(xs)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]

