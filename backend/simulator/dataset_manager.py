from __future__ import annotations

from typing import Dict


class DatasetManager:
    def __init__(self) -> None:
        self._datasets: Dict[str, dict] = {}

    def add(self, dataset_id: str, data: dict) -> None:
        self._datasets[dataset_id] = data

    def get(self, dataset_id: str) -> dict:
        if dataset_id not in self._datasets:
            raise KeyError("Dataset not found")
        return self._datasets[dataset_id]


dataset_manager = DatasetManager()

