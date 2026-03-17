from __future__ import annotations

from typing import Dict

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/simulator/assistant", tags=["simulator-assistant"])


class AssistantRequest(BaseModel):
    message: str
    context: Dict | None = None


@router.post("/query")
def assistant_query(req: AssistantRequest) -> Dict:
    message = req.message.strip()
    return {
        "reply": f"Assistant stub: received '{message}'.",
        "actions": [],
    }
