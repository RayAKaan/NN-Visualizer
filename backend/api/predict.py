from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.explanation import ExplanationBuilder
from services.inference import inference_engine

router = APIRouter()
explainer = ExplanationBuilder()


class PixelInput(BaseModel):
    pixels: list[float]


@router.post("/predict")
def predict(data: PixelInput) -> dict:
    try:
        result = inference_engine.predict(data.pixels)
        explanation = explainer.build(
            result["activations"],
            inference_engine.get_weights(),
            result["prediction"],
            result["probabilities"],
            result["input"],
        )
        return {**result, "explanation": explanation}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
