from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.session_manager import session_manager
from simulator.embeddings.embedding_engine import compute_embeddings

router = APIRouter(prefix="/api/simulator/embeddings", tags=["simulator-embeddings"])


class EmbeddingRequest(BaseModel):
    graph_id: str
    dataset_id: str
    layer_index: int
    method: str = "pca"
    n_samples: int = 200


@router.post("/compute")
def embeddings_compute(req: EmbeddingRequest) -> Dict:
    try:
        graph = session_manager.get_graph(req.graph_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return compute_embeddings(graph, req.dataset_id, req.layer_index, req.n_samples, req.method)
