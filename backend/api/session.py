from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .device import device_manager, get_device_info


@dataclass
class SessionModel:
    """Model definition in a session."""
    architecture: List[Dict] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    built: bool = False
    graph_id: Optional[str] = None


@dataclass
class SessionExecution:
    """Execution state in a session."""
    device: str = "cpu"
    status: str = "idle"  # idle, running, complete, error
    current_operation: Optional[str] = None  # forward, backward, training
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    error_message: Optional[str] = None


@dataclass
class SessionResults:
    """Execution results in a session."""
    activations: List = field(default_factory=list)
    gradients: Dict = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    profiling: Dict = field(default_factory=dict)
    layer_outputs: Dict = field(default_factory=dict)


@dataclass
class SessionDataset:
    """Dataset state in a session."""
    loaded: bool = False
    name: str = ""
    dataset_id: Optional[str] = None
    train_samples: int = 0
    test_samples: int = 0
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)


@dataclass
class SessionUI:
    """UI state in a session."""
    active_tab: str = "build"
    selected_layer: Optional[int] = None
    mode: str = "standard"  # beginner, standard, research


class SimulationSession:
    """Central state manager for a simulation session."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.updated_at = time.time()
        
        self.model = SessionModel()
        self.execution = SessionExecution()
        self.results = SessionResults()
        self.dataset = SessionDataset()
        self.ui = SessionUI()

    def build_model(self, architecture: List[Dict], hyperparameters: Dict) -> Dict:
        """Build the model with given architecture."""
        self.model.architecture = architecture
        self.model.hyperparameters = hyperparameters
        self.model.built = True
        self.updated_at = time.time()
        
        return {
            "success": True,
            "session_id": self.session_id,
            "architecture": architecture,
            "hyperparameters": hyperparameters,
        }

    def set_dataset(self, dataset_info: Dict) -> None:
        """Load dataset into session."""
        self.dataset.loaded = True
        self.dataset.name = dataset_info.get("name", "Unknown")
        self.dataset.dataset_id = dataset_info.get("dataset_id")
        self.dataset.train_samples = dataset_info.get("train_samples", 0)
        self.dataset.test_samples = dataset_info.get("test_samples", 0)
        self.dataset.input_shape = dataset_info.get("input_shape", [])
        self.dataset.output_shape = dataset_info.get("output_shape", [])
        self.updated_at = time.time()

    def start_execution(self, operation: str, total_epochs: int = 0) -> None:
        """Start an execution operation."""
        self.execution.status = "running"
        self.execution.current_operation = operation
        self.execution.current_epoch = 0
        self.execution.total_epochs = total_epochs
        self.execution.progress = 0.0
        self.execution.error_message = None
        self.updated_at = time.time()

    def update_progress(self, epoch: int, progress: float) -> None:
        """Update execution progress."""
        self.execution.current_epoch = epoch
        self.execution.progress = progress
        self.updated_at = time.time()

    def complete_execution(self, results: Dict) -> None:
        """Mark execution as complete."""
        self.execution.status = "complete"
        self.execution.progress = 1.0
        if results:
            self.results.metrics.update(results.get("metrics", {}))
            self.results.profiling.update(results.get("profiling", {}))
        self.updated_at = time.time()

    def fail_execution(self, error_message: str) -> None:
        """Mark execution as failed."""
        self.execution.status = "error"
        self.execution.error_message = error_message
        self.updated_at = time.time()

    def reset_execution(self) -> None:
        """Reset execution state."""
        self.execution.status = "idle"
        self.execution.current_operation = None
        self.execution.current_epoch = 0
        self.execution.progress = 0.0
        self.execution.error_message = None
        self.updated_at = time.time()

    def set_active_tab(self, tab: str) -> None:
        """Set active tab."""
        self.ui.active_tab = tab
        self.updated_at = time.time()

    def set_selected_layer(self, layer_index: Optional[int]) -> None:
        """Set selected layer."""
        self.ui.selected_layer = layer_index
        self.updated_at = time.time()

    def set_mode(self, mode: str) -> None:
        """Set user mode."""
        if mode in ["beginner", "standard", "research"]:
            self.ui.mode = mode
            self.updated_at = time.time()

    def get_state(self) -> Dict:
        """Get complete session state."""
        device_info = get_device_info()
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": {
                "built": self.model.built,
                "graph_id": self.model.graph_id,
                "architecture": self.model.architecture,
                "hyperparameters": self.model.hyperparameters,
            },
            "execution": {
                "device": self.execution.device or device_info.get("type", "cpu"),
                "status": self.execution.status,
                "current_operation": self.execution.current_operation,
                "current_epoch": self.execution.current_epoch,
                "total_epochs": self.execution.total_epochs,
                "progress": self.execution.progress,
                "error_message": self.execution.error_message,
            },
            "dataset": {
                "loaded": self.dataset.loaded,
                "name": self.dataset.name,
                "dataset_id": self.dataset.dataset_id,
                "train_samples": self.dataset.train_samples,
                "test_samples": self.dataset.test_samples,
                "input_shape": self.dataset.input_shape,
                "output_shape": self.dataset.output_shape,
            },
            "results": {
                "metrics": self.results.metrics,
                "profiling": self.results.profiling,
            },
            "ui": {
                "active_tab": self.ui.active_tab,
                "selected_layer": self.ui.selected_layer,
                "mode": self.ui.mode,
            },
            "device": device_info,
        }

    def get_status_summary(self) -> Dict:
        """Get a summary of session status."""
        device_info = get_device_info()
        
        return {
            "model_ready": self.model.built,
            "dataset_ready": self.dataset.loaded,
            "execution_status": self.execution.status,
            "device": device_info.get("type", "cpu"),
            "device_name": device_info.get("name", "CPU"),
            "next_action": self._get_next_action(),
        }

    def _get_next_action(self) -> str:
        """Determine next recommended action."""
        if not self.model.built:
            return "Build your model in the BUILD tab"
        if not self.dataset.loaded:
            return "Select a dataset in the BUILD tab"
        if self.execution.status == "idle":
            return "Click 'Forward' to run your model"
        if self.execution.status == "complete":
            if not self.results.gradients:
                return "Run backward pass to compute gradients"
            return "Inspect results in the INSPECT tab"
        return "Execution in progress..."


class SessionManager:
    """Manages multiple simulation sessions."""

    def __init__(self):
        self._sessions: Dict[str, SimulationSession] = {}
        self._current_session_id: Optional[str] = None

    def create_session(self) -> SimulationSession:
        """Create a new session."""
        session = SimulationSession()
        self._sessions[session.session_id] = session
        self._current_session_id = session.session_id
        return session

    def get_session(self, session_id: Optional[str] = None) -> Optional[SimulationSession]:
        """Get a session by ID."""
        if session_id is None:
            session_id = self._current_session_id
        return self._sessions.get(session_id)

    def get_or_create_session(self) -> SimulationSession:
        """Get current session or create new one."""
        if self._current_session_id and self._current_session_id in self._sessions:
            return self._sessions[self._current_session_id]
        return self.create_session()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            if self._current_session_id == session_id:
                self._current_session_id = None
            return True
        return False

    def set_current_session(self, session_id: str) -> bool:
        """Set current session."""
        if session_id in self._sessions:
            self._current_session_id = session_id
            return True
        return False

    def get_all_sessions(self) -> List[Dict]:
        """Get all session summaries."""
        return [
            {
                "session_id": sid,
                "created_at": session.created_at,
                "model_built": session.model.built,
                "dataset_loaded": session.dataset.loaded,
                "status": session.execution.status,
            }
            for sid, session in self._sessions.items()
        ]


session_manager = SessionManager()


def get_or_create_session() -> SimulationSession:
    """Get or create the current session."""
    return session_manager.get_or_create_session()


def get_session_status() -> Dict:
    """Get current session status."""
    session = get_or_create_session()
    return session.get_status_summary()


def get_session_state() -> Dict:
    """Get complete session state."""
    session = get_or_create_session()
    return session.get_state()