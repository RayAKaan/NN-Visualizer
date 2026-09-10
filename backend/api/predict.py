from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model_registry.base import ModelLoadError
from services.prediction_service import (
    ModelNotFoundError,
    ModelUnavailableError,
    prediction_service,
)

router = APIRouter()


class PredictRequest(BaseModel):
    model_id: str | None = None
    # Legacy family switch.  Two behaviours are supported for backwards
    # compatibility:
    #   * with pixels and no model_id  -> the original NN-Visualizer engine is
    #     used (legacy response incl. per-layer activations / explanation)
    #   * with model_id                -> registry adapter for that model
    model_type: str | None = None
    # Family-appropriate input — exactly one should be present depending on the
    # selected model's input_type:
    pixels: list[float] | None = None   # mnist_pixels (784 floats)
    text: str | None = None             # text
    image: str | None = None            # image (base64 / data URL)


def _legacy_predict(req: PredictRequest) -> dict:
    """Original /predict behaviour for legacy model_type callers.

    Kept so the older Prediction comparison UI and any training-mode clients
    continue receiving the exact historical payload (probabilities, per-layer
    activations, explanation).  Raises HTTP errors consistent with the rest of
    the API when the requested legacy weights are absent.
    """
    from services.explanation import explainer
    from services.inference import inference_engine

    if not req.pixels:
        raise HTTPException(status_code=400, detail="'pixels' is required for legacy MNIST models")
    model_type = (req.model_type or inference_engine.active_model_type).lower()
    if model_type not in inference_engine.models:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Legacy '{model_type}' weights are not loaded. Run the project training "
                "script (backend/train_*.py) or use model_id with a pretrained registry model."
            ),
        )
    result = inference_engine.predict(req.pixels, model_type)
    try:
        weights = inference_engine.get_weights(result["model_type"])
        result["explanation"] = explainer.build(result, weights, result["model_type"])
    except Exception:
        result["explanation"] = None
    return result


@router.post("/predict")
def predict(req: PredictRequest):
    # Backwards-compatible legacy route: model_type without a model_id.
    if req.model_id is None and req.model_type:
        return _legacy_predict(req)

    payload = {
        "model_id": req.model_id,
        "model_type": req.model_type,
        "pixels": req.pixels,
        "text": req.text,
        "image": req.image,
    }
    try:
        return prediction_service.predict(payload)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelLoadError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
