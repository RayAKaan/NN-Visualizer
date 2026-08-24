from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np


class DeviceManager:
    """Manages GPU/CPU device selection and memory."""

    def __init__(self):
        self._device_type: str = "cpu"
        self._device_name: str = "CPU"
        self._memory_total: int = 0
        self._memory_available: int = 0
        self._detect_device()

    def _detect_device(self) -> None:
        """Auto-detect best available device."""
        try:
            import torch
            
            if torch.cuda.is_available():
                self._device_type = "cuda"
                props = torch.cuda.get_device_properties(0)
                self._device_name = props.name
                self._memory_total = props.total_memory
                self._memory_available = self._memory_total - torch.cuda.memory_allocated(0)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device_type = "mps"
                self._device_name = "Apple Silicon GPU"
                self._memory_total = 8 * 1024 * 1024 * 1024  # Estimate 8GB
                self._memory_available = self._memory_total // 2
            else:
                self._device_type = "cpu"
                self._device_name = "CPU"
                self._memory_total = 16 * 1024 * 1024 * 1024  # Estimate 16GB
                self._memory_available = 8 * 1024 * 1024 * 1024
        except ImportError:
            self._device_type = "cpu"
            self._device_name = "CPU"
            self._memory_total = 16 * 1024 * 1024 * 1024
            self._memory_available = 8 * 1024 * 1024 * 1024

    @property
    def device(self) -> str:
        return self._device_type

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def memory_total(self) -> int:
        return self._memory_total

    @property
    def memory_available(self) -> int:
        return self._memory_available

    @property
    def memory_used(self) -> int:
        return self._memory_total - self._memory_available

    @property
    def memory_usage_percent(self) -> float:
        if self._memory_total == 0:
            return 0.0
        return (self.memory_used / self._memory_total) * 100

    def get_device_info(self) -> Dict:
        """Return device capabilities."""
        return {
            "type": self._device_type,
            "name": self._device_name,
            "memory_total": self._memory_total,
            "memory_available": self._memory_available,
            "memory_used": self.memory_used,
            "memory_usage_percent": self.memory_usage_percent,
            "cuda_available": self._is_cuda_available(),
            "mps_available": self._is_mps_available(),
        }

    def _is_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _is_mps_available(self) -> bool:
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except:
            return False

    def select_device(self, preference: str) -> Dict:
        """Allow manual device selection."""
        if preference == "gpu":
            if self._is_cuda_available():
                self._device_type = "cuda"
            elif self._is_mps_available():
                self._device_type = "mps"
            else:
                return {
                    "success": False,
                    "error": "GPU not available",
                    "message": "No GPU found. Falling back to CPU.",
                    "device": self._device_type,
                    "device_name": self._device_name,
                }
        elif preference == "cpu":
            self._device_type = "cpu"
            self._device_name = "CPU"

        return {
            "success": True,
            "device": self._device_type,
            "device_name": self._device_name,
            "message": f"Using {self._device_name}",
        }

    def clear_cache(self) -> Dict:
        """Clear GPU cache."""
        try:
            import torch
            if self._device_type == "cuda":
                torch.cuda.empty_cache()
                self._memory_available = self._memory_total - torch.cuda.memory_allocated(0)
                return {"success": True, "message": "GPU cache cleared"}
            elif self._device_type == "mps":
                torch.mps.empty_cache()
                return {"success": True, "message": "MPS cache cleared"}
        except:
            pass
        return {"success": True, "message": "No cache to clear"}

    def to_numpy(self, tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.detach().cpu().numpy()
        except:
            pass
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)


device_manager = DeviceManager()


def get_device_info() -> Dict:
    """Get current device information."""
    return device_manager.get_device_info()


def select_device(preference: str) -> Dict:
    """Select device manually."""
    return device_manager.select_device(preference)


def clear_device_cache() -> Dict:
    """Clear device cache."""
    return device_manager.clear_cache()


def get_memory_stats() -> Dict:
    """Get current memory statistics."""
    return {
        "total": device_manager.memory_total,
        "available": device_manager.memory_available,
        "used": device_manager.memory_used,
        "usage_percent": device_manager.memory_usage_percent,
    }