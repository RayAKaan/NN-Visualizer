from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import uuid
import numpy as np

from ..dataset_manager import dataset_manager
from ..graph_engine import NetworkGraph
from ..loss_functions import compute_loss
from .direction_methods import apply_vector_to_weights, param_vector_from_weights, random_directions


@dataclass
class LandscapeTask:
    task_id: str
    status: str
    progress: float
    result: Dict | None = None


class LandscapeManager:
    def __init__(self) -> None:
        self._tasks: Dict[str, LandscapeTask] = {}

    def compute(
        self,
        graph: NetworkGraph,
        dataset_id: str,
        resolution: int = 30,
        span: float = 1.0,
        seed: int | None = None,
    ) -> LandscapeTask:
        task_id = str(uuid.uuid4())
        task = LandscapeTask(task_id=task_id, status="computing", progress=0.0)
        self._tasks[task_id] = task

        dataset = dataset_manager.get(dataset_id)
        train = dataset.get("train", [])
        if isinstance(train, list):
            data = [(np.asarray(p["x"], dtype=np.float32), np.asarray(p["y"], dtype=np.float32)) for p in train]
        else:
            xs = np.asarray(train.get("x", []), dtype=np.float32)
            ys = np.asarray(train.get("y", []), dtype=np.float32)
            data = [(xs[i], ys[i]) for i in range(len(xs))]

        base_vec = param_vector_from_weights(graph.weights, graph.biases)
        d1, d2 = random_directions(base_vec, seed=seed)
        grid = np.linspace(-span, span, resolution)
        loss_surface = np.zeros((resolution, resolution), dtype=np.float32)

        def _set_graph_params(weights: List[np.ndarray], biases: List[np.ndarray]) -> None:
            graph.weights = [w.astype(np.float32) for w in weights]
            graph.biases = [b.astype(np.float32) for b in biases]
            for layer_idx, param_idx in graph.param_index_by_layer.items():
                layer = graph.layer_instances[layer_idx]
                if hasattr(layer, "set_params"):
                    layer.set_params(graph.weights[param_idx], graph.biases[param_idx])

        original_weights = [w.copy() for w in graph.weights]
        original_biases = [b.copy() for b in graph.biases]

        total = resolution * resolution
        counter = 0
        try:
            for i, a in enumerate(grid):
                for j, b in enumerate(grid):
                    new_w, new_b = apply_vector_to_weights(graph.weights, graph.biases, d1, d2, float(a), float(b))
                    _set_graph_params(new_w, new_b)
                    loss_val = 0.0
                    for x, y in data:
                        pred = graph.forward(x)
                        loss_val += compute_loss(y, pred, "mse")[0]
                    loss_val /= max(len(data), 1)
                    loss_surface[i, j] = loss_val
                    counter += 1
                    task.progress = counter / total
        finally:
            _set_graph_params(original_weights, original_biases)

        task.status = "complete"
        task.result = {
            "grid_x": grid.tolist(),
            "grid_y": grid.tolist(),
            "loss_surface": loss_surface.tolist(),
            "center_point": [0.0, 0.0],
            "center_loss": float(loss_surface[resolution // 2, resolution // 2]),
            "min_loss": float(loss_surface.min()),
            "min_location": [float(grid[np.unravel_index(np.argmin(loss_surface), loss_surface.shape)[0]]), float(grid[np.unravel_index(np.argmin(loss_surface), loss_surface.shape)[1]])],
            "max_loss": float(loss_surface.max()),
            "critical_points": [],
            "training_trajectory": [],
            "sharpness_score": float(loss_surface.max() - loss_surface.min()),
            "flatness_description": "rough",
        }
        return task

    def get(self, task_id: str) -> LandscapeTask:
        if task_id not in self._tasks:
            raise KeyError("Landscape task not found")
        return self._tasks[task_id]


landscape_manager = LandscapeManager()
