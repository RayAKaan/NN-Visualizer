from fastapi import APIRouter, HTTPException
import time

from .device import get_device_info, select_device as ds_select_device, clear_device_cache, get_memory_stats
from .response import success_response, error_response, success_response as res_success

router = APIRouter(prefix="/api/device", tags=["device"])


@router.get("/info")
def device_info():
    """Get current device information."""
    start_time = time.time()
    info = get_device_info()
    return res_success(
        data=info,
        execution_time=time.time() - start_time,
        device=info.get("type", "cpu"),
    )


@router.post("/select")
def device_select(preference: str = "auto"):
    """Select device manually (auto, gpu, cpu)."""
    start_time = time.time()
    result = ds_select_device(preference)
    
    if result.get("success"):
        return res_success(
            data=result,
            execution_time=time.time() - start_time,
            device=result.get("device", "cpu"),
        )
    else:
        return error_response(
            code="DEVICE_SELECTION_FAILED",
            message=result.get("message", "Failed to select device"),
            details={"preference": preference},
        )


@router.post("/clear_cache")
def device_clear_cache():
    """Clear device cache."""
    start_time = time.time()
    result = clear_device_cache()
    info = get_device_info()
    
    return res_success(
        data=result,
        execution_time=time.time() - start_time,
        device=info.get("type", "cpu"),
    )


@router.get("/memory")
def device_memory():
    """Get memory statistics."""
    start_time = time.time()
    stats = get_memory_stats()
    info = get_device_info()
    
    return res_success(
        data={
            **stats,
            "device_name": info.get("name", "CPU"),
        },
        execution_time=time.time() - start_time,
        device=info.get("type", "cpu"),
    )