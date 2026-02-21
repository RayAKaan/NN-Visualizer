from __future__ import annotations

from fastapi import APIRouter, HTTPException

from services.inference import inference_engine

router = APIRouter()


@router.get("/")
def root() -> dict:
    return {"status": "ok", "model_loaded": inference_engine.is_loaded}


@router.get("/health")
def health() -> dict:
    return inference_engine.health()


@router.get("/model/info")
def model_info() -> dict:
    try:
        return inference_engine.get_model_info()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/samples")
def samples() -> dict[str, list[float]]:
    return inference_engine.get_samples()
