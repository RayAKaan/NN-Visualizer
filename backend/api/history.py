from fastapi import APIRouter
from training.manager import training_manager

router = APIRouter()


@router.get("/training/history")
def get_training_history():
    """Return all stored batch snapshots for replay."""
    return training_manager.get_training_history()


@router.get("/training/epochs")
def get_epoch_history():
    """Return epoch summary history."""
    return training_manager.get_epoch_history()


@router.get("/training/status")
def get_training_status():
    """Return current training status and progress."""
    return training_manager.get_status()
