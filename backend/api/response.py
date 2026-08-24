from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import Request
from starlette.datastructures import Headers


@dataclass
class APIResponseMeta:
    """Standard metadata for API responses."""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    execution_time: float = 0.0
    device: str = "cpu"
    gpu_memory_used: Optional[int] = None
    request_id: Optional[str] = None


@dataclass 
class APIError:
    """Standard error format."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


def create_response(
    data: Any = None,
    status: str = "success",
    error: Optional[APIError] = None,
    execution_time: float = 0.0,
    device: str = "cpu",
    gpu_memory_used: Optional[int] = None,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    """Create a standard API response."""
    
    # Get request ID if available
    request_id = None
    if request and hasattr(request, 'state') and hasattr(request.state, 'request_id'):
        request_id = request.state.request_id
    
    meta = APIResponseMeta(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        execution_time=execution_time,
        device=device,
        gpu_memory_used=gpu_memory_used,
        request_id=request_id,
    )
    
    return {
        "status": status,
        "data": data,
        "meta": {
            "timestamp": meta.timestamp,
            "execution_time": meta.execution_time,
            "device": meta.device,
            "gpu_memory_used": meta.gpu_memory_used,
            "request_id": meta.request_id,
        },
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        } if error else None,
    }


def success_response(
    data: Any,
    execution_time: float = 0.0,
    device: str = "cpu",
    gpu_memory_used: Optional[int] = None,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    """Create a success response."""
    return create_response(
        data=data,
        status="success",
        execution_time=execution_time,
        device=device,
        gpu_memory_used=gpu_memory_used,
        request=request,
    )


def error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    """Create an error response."""
    error = APIError(code=code, message=message, details=details)
    return create_response(
        data=None,
        status="error",
        error=error,
        request=request,
    )


def device_response(device_info: Dict) -> Dict[str, Any]:
    """Create a device info response."""
    return success_response(data=device_info)


def oom_error_response(
    required: int,
    available: int,
    suggestion: str = "Reduce batch size or use CPU",
) -> Dict[str, Any]:
    """Create an out-of-memory error response."""
    return error_response(
        code="CUDA_OUT_OF_MEMORY",
        message="GPU memory exceeded",
        details={
            "required": required,
            "available": available,
            "suggestion": suggestion,
        },
    )


def validation_error_response(
    message: str,
    field_errors: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Create a validation error response."""
    return error_response(
        code="VALIDATION_ERROR",
        message=message,
        details=field_errors,
    )


def not_found_error_response(
    resource: str,
    resource_id: str,
) -> Dict[str, Any]:
    """Create a not found error response."""
    return error_response(
        code="NOT_FOUND",
        message=f"{resource} '{resource_id}' not found",
        details={"resource": resource, "id": resource_id},
    )


def unsupported_device_error_response(
    device: str,
    supported: list,
) -> Dict[str, Any]:
    """Create an unsupported device error response."""
    return error_response(
        code="UNSUPPORTED_DEVICE",
        message=f"Device '{device}' not supported",
        details={
            "requested": device,
            "supported": supported,
        },
    )