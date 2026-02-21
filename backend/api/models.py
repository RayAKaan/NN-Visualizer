from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import MODELS_DIR
from services.inference import inference_engine
from training.manager import training_manager

router = APIRouter()


class ModelNameInput(BaseModel):
    name: str


@router.get("/models")
def list_models() -> dict:
    return {"models": training_manager.list_models()}


@router.post("/models/save")
def save_model(data: ModelNameInput) -> dict:
    try:
        path = training_manager.save_latest(data.name)
        return {"saved": True, "path": path}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{name}")
def delete_model(name: str) -> dict:
    training_manager.delete_model(name)
    return {"deleted": True, "name": name}


@router.post("/models/load")
def load_saved_model(data: ModelNameInput) -> dict:
    path = MODELS_DIR / f"{data.name}.keras"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Saved model not found: {data.name}")
    inference_engine.load(str(path))
    return {"loaded": True, "name": data.name}
