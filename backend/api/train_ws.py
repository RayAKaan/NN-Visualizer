from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from training.manager import TrainingConfig, training_manager

router = APIRouter()


class WebSocketManager:
    def __init__(self) -> None:
        self.connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        for connection in list(self.connections):
            try:
                await connection.send_json(message)
            except RuntimeError:
                self.disconnect(connection)


websocket_manager = WebSocketManager()


async def send_training_message(message: dict) -> None:
    await websocket_manager.broadcast(message)


@router.websocket("/train")
async def training_ws(websocket: WebSocket) -> None:
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            if command == "configure":
                training_manager.configure(
                    TrainingConfig(
                        learning_rate=float(data.get("learning_rate", 0.001)),
                        batch_size=int(data.get("batch_size", 64)),
                        epochs=int(data.get("epochs", 5)),
                        optimizer=str(data.get("optimizer", "adam")).lower(),
                        weight_decay=float(data.get("weight_decay", 0.0)),
                        activation=str(data.get("activation", "relu")).lower(),
                        initializer=str(data.get("initializer", "glorot_uniform")),
                        dropout=float(data.get("dropout", 0.0)),
                    )
                )
                await websocket.send_json({"type": "status", "status": "configured"})
            elif command == "start":
                try:
                    training_manager.start(send_training_message, asyncio.get_running_loop())
                    await websocket.send_json({"type": "status", "status": training_manager.status()})
                except RuntimeError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
            elif command == "pause":
                training_manager.pause()
                await websocket.send_json({"type": "status", "status": training_manager.status()})
            elif command == "resume":
                training_manager.resume()
                await websocket.send_json({"type": "status", "status": training_manager.status()})
            elif command == "stop":
                training_manager.stop()
                await websocket.send_json({"type": "status", "status": training_manager.status()})
            elif command == "step_batch":
                training_manager.step_batch()
            elif command == "step_epoch":
                training_manager.step_epoch()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@router.get("/training/status")
def training_status() -> dict:
    return {
        "status": training_manager.status(),
        "running": training_manager.is_running(),
        "metrics": training_manager.latest_metrics(),
    }
