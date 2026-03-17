from __future__ import annotations

from typing import Dict, List
import json
import uuid
from pathlib import Path


class ExperimentManager:
    def __init__(self) -> None:
        self._path = Path("models_store") / "experiments.json"
        self._store: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._store = {}
            return
        try:
            self._store = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            self._store = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def create(self, payload: Dict) -> Dict:
        exp_id = str(uuid.uuid4())
        record = {"id": exp_id, **payload}
        self._store[exp_id] = record
        self._save()
        return record

    def list(self) -> List[Dict]:
        return list(self._store.values())

    def get(self, exp_id: str) -> Dict:
        if exp_id not in self._store:
            raise KeyError("Experiment not found")
        return self._store[exp_id]

    def delete(self, exp_id: str) -> None:
        if exp_id in self._store:
            del self._store[exp_id]
            self._save()


experiment_manager = ExperimentManager()
