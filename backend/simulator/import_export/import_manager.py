from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import uuid


@dataclass
class ImportArtifact:
    import_id: str
    format: str
    architecture: Dict
    warnings: List[str]


class ImportManager:
    def __init__(self) -> None:
        self._imports: Dict[str, ImportArtifact] = {}

    def add(self, format: str, architecture: Dict, warnings: List[str]) -> ImportArtifact:
        import_id = str(uuid.uuid4())
        artifact = ImportArtifact(import_id=import_id, format=format, architecture=architecture, warnings=warnings)
        self._imports[import_id] = artifact
        return artifact

    def get(self, import_id: str) -> ImportArtifact:
        if import_id not in self._imports:
            raise KeyError("Import not found")
        return self._imports[import_id]


import_manager = ImportManager()
