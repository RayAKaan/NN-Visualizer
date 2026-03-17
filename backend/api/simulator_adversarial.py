from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.session_manager import session_manager
from simulator.adversarial import fgsm_attack, pgd_attack, evaluate_robustness

router = APIRouter(prefix="/api/simulator/adversarial", tags=["simulator-adversarial"])


class AttackRequest(BaseModel):
    graph_id: str
    input: List[float]
    true_label: int
    method: str = "fgsm"
    epsilon: float = 0.1
    input_shape: List[int] | None = None
    pgd_steps: int = 7
    pgd_step_size: float = 0.02


class EvaluateRequest(BaseModel):
    graph_id: str
    dataset_id: str
    epsilons: List[float]
    pgd_steps: int = 5


@router.post("/attack")
def adversarial_attack(req: AttackRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    method = (req.method or "fgsm").lower()
    if method == "pgd":
        return pgd_attack(
            graph,
            req.input,
            req.true_label,
            req.epsilon,
            req.pgd_step_size,
            req.pgd_steps,
            req.input_shape,
        )
    return fgsm_attack(graph, req.input, req.true_label, req.epsilon, req.input_shape)


@router.post("/evaluate")
def adversarial_evaluate(req: EvaluateRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from simulator.dataset_manager import dataset_manager

    dataset = dataset_manager.get(req.dataset_id)
    train = dataset.get("train", [])
    samples = []
    if isinstance(train, list):
        for item in train[:100]:
            x = item["x"]
            y = item["y"]
            label = int(max(range(len(y)), key=lambda i: y[i])) if hasattr(y, '__len__') else int(y)
            samples.append((x, label))
    else:
        xs = train.get("x", [])
        ys = train.get("y", [])
        for i in range(min(len(xs), 100)):
            y = ys[i]
            label = int(max(range(len(y)), key=lambda j: y[j])) if hasattr(y, '__len__') else int(y)
            samples.append((xs[i], label))

    return evaluate_robustness(graph, [(s[0], s[1]) for s in samples], req.epsilons, req.pgd_steps)
