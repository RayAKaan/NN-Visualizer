from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.session_manager import session_manager
from simulator.landscape.surface_computer import landscape_manager

router = APIRouter(prefix="/api/simulator/landscape", tags=["simulator-landscape"])


class LandscapeRequest(BaseModel):
    graph_id: str
    dataset_id: str
    method: str = "random"
    resolution: int = 30
    range: float = 1.0


@router.post("/compute")
def landscape_compute(req: LandscapeRequest):
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    task = landscape_manager.compute(graph, req.dataset_id, req.resolution, req.range)
    return {
        "task_id": task.task_id,
        "status": task.status,
        "total_evaluations": req.resolution * req.resolution,
        "estimated_time_seconds": 0,
    }


@router.get("/status/{task_id}")
def landscape_status(task_id: str):
    try:
        task = landscape_manager.get(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "status": task.status,
        "progress": task.progress,
        "result": task.result,
    }
