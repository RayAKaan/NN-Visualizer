from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.inference import inference_engine

router = APIRouter()


class PixelInput(BaseModel):
    pixels: list[float]


@router.post("/state")
def state(data: PixelInput) -> dict:
    try:
        return inference_engine.get_state(data.pixels)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
