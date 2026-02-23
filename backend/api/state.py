from fastapi import APIRouter
from pydantic import BaseModel

from services.inference import inference_engine

router = APIRouter()


class StateRequest(BaseModel):
    pixels: list[float]
    model_type: str | None = None


@router.post("/state")
def state(req: StateRequest):
    return inference_engine.get_state(req.pixels, req.model_type)
