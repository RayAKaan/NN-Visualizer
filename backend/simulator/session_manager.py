from __future__ import annotations

import uuid
from typing import Dict, List

from .graph_engine import NetworkGraph, build_graph
from .layers import LayerConfig


class SessionManager:
    def __init__(self) -> None:
        self._graphs: Dict[str, NetworkGraph] = {}

    def create_graph(self, layers: List[LayerConfig]) -> str:
        graph = build_graph(layers)
        graph_id = str(uuid.uuid4())
        self._graphs[graph_id] = graph
        return graph_id

    def get_graph(self, graph_id: str) -> NetworkGraph:
        if graph_id not in self._graphs:
            raise KeyError("Graph not found")
        return self._graphs[graph_id]

    def delete_graph(self, graph_id: str) -> None:
        self._graphs.pop(graph_id, None)


session_manager = SessionManager()

