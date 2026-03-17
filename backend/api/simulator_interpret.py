from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.session_manager import session_manager
from simulator.interpretability.integrated_gradients import integrated_gradients
from simulator.interpretability.deep_shap import deep_shap
from simulator.interpretability.lime_explainer import lime_explain
from simulator.interpretability.lrp import lrp_explain
from simulator.interpretability.method_comparator import compare_methods

router = APIRouter(prefix="/api/simulator/interpret", tags=["simulator-interpret"])


class IGRequest(BaseModel):
    graph_id: str
    input: List[float]
    target_class: int = 0
    n_steps: int = 50


class ShapRequest(BaseModel):
    graph_id: str
    input: List[float]
    target_class: int = 0
    baseline_samples: int = 50


class LimeRequest(BaseModel):
    graph_id: str
    input: List[float]
    input_shape: List[int] | None = None
    target_class: int = 0
    n_samples: int = 200


class LrpRequest(BaseModel):
    graph_id: str
    input: List[float]
    target_class: int = 0
    rule: str = "epsilon"
    epsilon: float = 1e-2


class InterpretCompareRequest(BaseModel):
    graph_id: str
    input: List[float]
    target_class: int = 0
    methods: List[str] = ["gradient", "integrated_gradients", "shap", "lime", "lrp"]


@router.post("/integrated_gradients")
def interpret_integrated_gradients(req: IGRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return integrated_gradients(graph, req.input, req.target_class, req.n_steps)

@router.post("/shap")
def interpret_shap(req: ShapRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return deep_shap(graph, req.input, req.target_class, req.baseline_samples)


@router.post("/lime")
def interpret_lime(req: LimeRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return lime_explain(graph, req.input, req.input_shape, req.target_class, req.n_samples)


@router.post("/lrp")
def interpret_lrp(req: LrpRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return lrp_explain(graph, req.input, req.target_class, req.epsilon)


@router.post("/compare")
def interpret_compare(req: InterpretCompareRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return compare_methods(graph, req.input, req.target_class)

