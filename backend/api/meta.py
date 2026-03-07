import numpy as np
import tensorflow as tf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.inference import inference_engine
from training.manager import training_manager

router = APIRouter()


class SwitchRequest(BaseModel):
    model_type: str


@router.get("/health")
def health():
    return {
        "status": "ok" if inference_engine.loaded else "degraded",
        "loaded_models": inference_engine.get_available_models(),
        "active_model": inference_engine.active_model_type,
    }


@router.get("/model/info")
def model_info(type: str | None = None):
    return inference_engine.get_model_info(type)


@router.get("/models/available")
def models_available():
    return {"available": inference_engine.get_available_models(), "active": inference_engine.active_model_type}


@router.get("/models/registry")
def models_registry():
    return {
        "models": inference_engine.get_models_registry(),
        "available": inference_engine.get_available_models(),
        "active": inference_engine.active_model_type,
    }


@router.post("/model/switch")
def model_switch(req: SwitchRequest):
    try:
        inference_engine.set_active_model(req.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"active": inference_engine.active_model_type}


@router.post("/models/{model_type}/reload")
def model_reload(model_type: str):
    try:
        return inference_engine.reload_model(model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{model_type}")
def model_delete(model_type: str):
    try:
        return inference_engine.delete_model(model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/{model_type}/save")
def model_save(model_type: str):
    try:
        saved = training_manager.save_model(model_type)
        reload_info = inference_engine.reload_model(model_type)
        return {
            **saved,
            "reloaded": reload_info.get("reloaded", False),
            "available": reload_info.get("available", inference_engine.get_available_models()),
            "active": inference_engine.active_model_type,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/samples")
def samples():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    out = {}
    for digit in range(10):
        idx = int(np.where(y_test == digit)[0][0])
        out[str(digit)] = (x_test[idx].astype("float32") / 255.0).reshape(-1).tolist()
    return out
