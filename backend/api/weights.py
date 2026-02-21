from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.inference import inference_engine

router = APIRouter()


@router.get("/weights")
def weights() -> dict:
    try:
        return inference_engine.get_weights()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
