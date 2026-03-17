from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from simulator.generative.manager import generative_manager
from simulator.dataset_manager import dataset_manager

router = APIRouter(prefix="/api/simulator/generative", tags=["simulator-generative"])


class SampleRequest(BaseModel):
    mode: str = "vae"
    n_samples: int = 8
    size: int = 28


class TrainRequest(BaseModel):
    dataset_id: str
    mode: str = "vae"
    epochs: int = 5


@router.post("/train")
def generative_train(req: TrainRequest) -> Dict:
    dataset = dataset_manager.get(req.dataset_id)
    train = dataset.get("train", [])
    if isinstance(train, list):
        xs = [p["x"] for p in train]
    else:
        xs = train.get("x", [])
    import numpy as np
    x = np.asarray(xs, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((1, -1))
    return generative_manager.train(x, req.mode, req.epochs)


@router.post("/sample")
def generative_sample(req: SampleRequest) -> Dict:
    return generative_manager.sample(req.mode, req.n_samples, req.size)
