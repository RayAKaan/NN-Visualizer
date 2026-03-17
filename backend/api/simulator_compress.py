from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.session_manager import session_manager
from simulator.compression import magnitude_prune, quantize_weights, pruning_sweep

router = APIRouter(prefix="/api/simulator/compress", tags=["simulator-compress"])


class PruneRequest(BaseModel):
    graph_id: str
    method: str = "magnitude"
    sparsity: float = 0.5


class QuantizeRequest(BaseModel):
    graph_id: str
    target_dtype: str = "int8"


class SweepRequest(BaseModel):
    graph_id: str
    sparsity_range: List[float]


@router.post("/prune")
def compress_prune(req: PruneRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return magnitude_prune(graph, req.sparsity)


@router.post("/quantize")
def compress_quantize(req: QuantizeRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return quantize_weights(graph, req.target_dtype)


@router.post("/sweep")
def compress_sweep(req: SweepRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return pruning_sweep(graph, req.sparsity_range)
