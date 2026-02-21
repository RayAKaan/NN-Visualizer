from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.inference import inference_engine

router = APIRouter()


class ModelSwitchInput(BaseModel):
    model_type: str = "ann"


@router.get("/")
def root() -> dict:
    return {"status": "ok", "model_loaded": inference_engine.is_loaded}


@router.get("/health")
def health() -> dict:
    return inference_engine.health()


@router.get("/model/info")
def model_info(type: str | None = None) -> dict:
    try:
        return inference_engine.get_model_info(model_type=type)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/model/switch")
def switch_model(data: ModelSwitchInput) -> dict:
    try:
        inference_engine.set_active_model(data.model_type)
        return {
            "active_model": inference_engine.active_model_type,
            "available_models": inference_engine.get_available_models(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models/available")
def models_available() -> dict:
    return {
        "available": inference_engine.get_available_models(),
        "active": inference_engine.active_model_type,
    }


@router.get("/samples")
def samples(model_type: str | None = None) -> dict[str, list[float]]:
    return inference_engine.get_samples(model_type=model_type)
