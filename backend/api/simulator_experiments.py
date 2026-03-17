from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.experiments import experiment_manager

router = APIRouter(prefix="/api/simulator/experiments", tags=["simulator-experiments"])


class ExperimentCreateRequest(BaseModel):
    name: str
    config: Dict
    metrics: Dict | None = None
    notes: str | None = None
    tags: List[str] | None = None


@router.post("/create")
def experiments_create(req: ExperimentCreateRequest) -> Dict:
    return experiment_manager.create(req.model_dump())


@router.get("/list")
def experiments_list() -> Dict:
    return {"experiments": experiment_manager.list()}


@router.get("/{experiment_id}")
def experiments_get(experiment_id: str) -> Dict:
    try:
        return experiment_manager.get(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/{experiment_id}")
def experiments_delete(experiment_id: str) -> Dict:
    experiment_manager.delete(experiment_id)
    return {"deleted": True}
