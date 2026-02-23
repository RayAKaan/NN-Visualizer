from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.inference import inference_engine

router = APIRouter()


class StateInput(BaseModel):
    pixels: list[float]
    model_type: str | None = None


@router.post("/state")
def state(data: StateInput) -> dict:
    try:
        return inference_engine.get_state(data.pixels, model_type=data.model_type)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
