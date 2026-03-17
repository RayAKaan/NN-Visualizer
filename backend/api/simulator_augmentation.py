from __future__ import annotations

from typing import Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from simulator.augmentation import preview_augmentations

router = APIRouter(prefix="/api/simulator/augmentation", tags=["simulator-augmentation"])


class PreviewRequest(BaseModel):
    input: List[float]
    input_shape: List[int] | None = None
    n_samples: int = 8
    pipeline: List[Dict] | None = None


@router.post("/preview")
def augment_preview(req: PreviewRequest) -> Dict:
    return {"samples": preview_augmentations(req.input, req.input_shape, req.n_samples, req.pipeline)}
