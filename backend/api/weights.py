from fastapi import APIRouter

from services.inference import inference_engine

router = APIRouter()


@router.get("/weights")
def get_weights(model_type: str | None = None):
    return inference_engine.get_weights(model_type)
