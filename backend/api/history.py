from fastapi import APIRouter

from training.manager import training_manager

router = APIRouter()


@router.get("/training/history")
def training_history():
    return training_manager.get_training_history()


@router.get("/training/epochs")
def training_epochs():
    return training_manager.get_epoch_history()


@router.get("/training/status")
def training_status():
    return training_manager.get_status()
