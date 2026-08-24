from fastapi import APIRouter, HTTPException
import time

from .session import get_session_state, get_session_status, get_or_create_session
from .response import success_response, error_response

router = APIRouter(prefix="/api/session", tags=["session"])


@router.get("/state")
def session_state():
    """Get complete session state."""
    start_time = time.time()
    state = get_session_state()
    device_info = state.get("device", {})
    
    return success_response(
        data=state,
        execution_time=time.time() - start_time,
        device=device_info.get("type", "cpu"),
        gpu_memory_used=device_info.get("memory_used"),
    )


@router.get("/status")
def session_status():
    """Get session status summary."""
    start_time = time.time()
    status = get_session_status()
    device_info = status  # status includes device info
    
    return success_response(
        data=status,
        execution_time=time.time() - start_time,
        device=status.get("device", "cpu"),
    )


@router.post("/reset")
def session_reset():
    """Reset session execution state."""
    start_time = time.time()
    session = get_or_create_session()
    session.reset_execution()
    
    return success_response(
        data={"message": "Session reset successfully"},
        execution_time=time.time() - start_time,
    )


@router.post("/tab/{tab}")
def set_active_tab(tab: str):
    """Set active tab."""
    start_time = time.time()
    session = get_or_create_session()
    session.set_active_tab(tab)
    
    return success_response(
        data={"active_tab": tab},
        execution_time=time.time() - start_time,
    )


@router.post("/layer/{layer_index}")
def set_selected_layer(layer_index: int):
    """Set selected layer."""
    start_time = time.time()
    session = get_or_create_session()
    session.set_selected_layer(layer_index)
    
    return success_response(
        data={"selected_layer": layer_index},
        execution_time=time.time() - start_time,
    )


@router.post("/mode/{mode}")
def set_mode(mode: str):
    """Set user mode (beginner, standard, research)."""
    start_time = time.time()
    session = get_or_create_session()
    session.set_mode(mode)
    
    return success_response(
        data={"mode": mode},
        execution_time=time.time() - start_time,
    )