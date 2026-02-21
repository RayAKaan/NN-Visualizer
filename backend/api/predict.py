from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.explanation import ExplanationBuilder
from services.inference import inference_engine

router = APIRouter()
explainer = ExplanationBuilder()


class PixelInput(BaseModel):
    pixels: list[float]
    model_type: str | None = None


@router.post("/predict")
def predict(data: PixelInput) -> dict:
    try:
        mt = data.model_type or inference_engine.active_model_type
        result = inference_engine.predict(data.pixels, model_type=mt)
        if mt == "ann":
            result["explanation"] = explainer.build(
                result.get("activations", {}),
                inference_engine.weights_cache.get("ann", {}),
                result["prediction"],
                result["probabilities"],
                result.get("input"),
            )
        else:
            result["explanation"] = explainer.build_cnn(
                result.get("feature_maps", []),
                result.get("dense_layers", {}),
                result.get("prediction", -1),
                result.get("probabilities", []),
            )
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
