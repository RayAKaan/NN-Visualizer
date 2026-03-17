from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .graph_engine import NetworkGraph


@dataclass
class Snapshot:
    epoch: int
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    metrics: dict
    boundary: List[List[float]] | None = None


class SnapshotManager:
    def __init__(self) -> None:
        self._snapshots: Dict[str, List[Snapshot]] = {}

    def add_snapshot(self, graph_id: str, snapshot: Snapshot, max_snapshots: int = 200) -> None:
        if graph_id not in self._snapshots:
            self._snapshots[graph_id] = []
        self._snapshots[graph_id].append(snapshot)
        if len(self._snapshots[graph_id]) > max_snapshots:
            self._snapshots[graph_id] = self._snapshots[graph_id][-max_snapshots:]

    def list_snapshots(self, graph_id: str) -> List[Snapshot]:
        return self._snapshots.get(graph_id, [])

    def get_snapshot(self, graph_id: str, index: int) -> Snapshot:
        snaps = self._snapshots.get(graph_id, [])
        if index < 0 or index >= len(snaps):
            raise IndexError("snapshot_index out of range")
        return snaps[index]

    @staticmethod
    def apply_snapshot(graph: NetworkGraph, snapshot: Snapshot) -> None:
        graph.weights = [w.copy() for w in snapshot.weights]
        graph.biases = [b.copy() for b in snapshot.biases]
        if graph.layer_instances:
            for layer_idx, param_idx in graph.param_index_by_layer.items():
                layer = graph.layer_instances[layer_idx]
                layer.set_params(graph.weights[param_idx], graph.biases[param_idx])


snapshot_manager = SnapshotManager()
