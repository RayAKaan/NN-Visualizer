from fastapi import APIRouter
from pydantic import BaseModel

from services.inference import inference_engine
from services.explanation import explainer

router = APIRouter()


class PredictRequest(BaseModel):
    pixels: list[float]
    model_type: str | None = None


@router.post("/predict")
def predict(req: PredictRequest):
    result = inference_engine.predict(req.pixels, req.model_type)
    weights = inference_engine.get_weights(result["model_type"])
    result["explanation"] = explainer.build(result, weights, result["model_type"])
    return result
