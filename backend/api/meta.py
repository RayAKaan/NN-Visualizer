import numpy as np
import tensorflow as tf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.inference import inference_engine

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


@router.post("/model/switch")
def model_switch(req: SwitchRequest):
    try:
        inference_engine.set_active_model(req.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"active": inference_engine.active_model_type}


@router.get("/samples")
def samples():
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    out = {}
    for digit in range(10):
        idx = int(np.where(y_test == digit)[0][0])
        out[str(digit)] = (x_test[idx].astype("float32") / 255.0).reshape(-1).tolist()
    return out
