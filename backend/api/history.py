from __future__ import annotations

from fastapi import APIRouter

from training.manager import training_manager

router = APIRouter()


@router.get("/training/history")
def get_training_history() -> list[dict]:
    return training_manager.get_training_history()
