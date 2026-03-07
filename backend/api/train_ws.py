import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from training.manager import training_manager

router = APIRouter()


@router.websocket("/train")
async def train_ws(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    async def _send_status():
        current = training_manager.get_status()
        status_map = {
            "running": "training",
            "paused": "paused",
            "completed": "completed",
            "stopped": "stopped",
            "error": "idle",
            "idle": "idle",
        }
        await ws.send_json(
            {
                "type": "status",
                "status": status_map.get(current.get("status", "idle"), "idle"),
                "current_epoch": current.get("epoch", 0),
                "total_epochs": current.get("config", {}).get("epochs", 0),
            }
        )

    def _send_threadsafe(message: dict):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(message), loop)
        except RuntimeError:
            # Event loop is not available (shutdown/disconnect in progress).
            return

    def emit(payload):
        ptype = payload.get("type")

        if ptype == "batch_update":
            msg = {
                "type": "batch",
                "epoch": payload.get("epoch", 0),
                "batch": payload.get("batch", 0),
                "total_batches": payload.get("total_batches", 0),
                "total_epochs": payload.get("total_epochs", 0),
                "model_type": payload.get("model_type"),
                "loss": payload.get("loss", 0.0),
                "accuracy": payload.get("accuracy", 0.0),
                "learning_rate": payload.get("learning_rate", 0.0),
                "gradient_norm": payload.get("gradient_norm", 0.0),
                "timestamp": payload.get("timestamp"),
            }
            _send_threadsafe(msg)
            return

        if ptype == "epoch_update":
            msg = {
                "type": "epoch",
                "epoch": payload.get("epoch", 0),
                "total_epochs": payload.get("total_epochs", 0),
                "loss": payload.get("loss", 0.0),
                "accuracy": payload.get("accuracy", 0.0),
                "val_loss": payload.get("val_loss", 0.0),
                "val_accuracy": payload.get("val_accuracy", 0.0),
                "precision_per_class": payload.get("precision_per_class", []),
                "recall_per_class": payload.get("recall_per_class", []),
                "f1_per_class": payload.get("f1_per_class", []),
                "confusion_matrix": payload.get("confusion_matrix", []),
            }
            _send_threadsafe(msg)
            return

        if ptype == "training_complete":
            history = training_manager.get_epoch_history()
            last = history[-1] if history else {}
            msg = {
                "type": "complete",
                "test_accuracy": float(last.get("val_accuracy", 0.0)),
            }
            _send_threadsafe(msg)
            return

        if ptype == "training_error":
            msg = {"type": "error", "message": payload.get("error", "Training error")}
            _send_threadsafe(msg)
            return

        if ptype == "training_stopped":
            _send_threadsafe({"type": "status", "status": "stopped"})
            return

        _send_threadsafe(payload)

    try:
        while True:
            data = json.loads(await ws.receive_text())
            cmd = data.get("command") or data.get("action")
            if cmd == "configure":
                training_manager.configure(data.get("config", {}))
                await _send_status()
            elif cmd == "start":
                cfg = data.get("config")
                if isinstance(cfg, dict):
                    training_manager.configure(cfg)
                training_manager.start(emit)
                await ws.send_json({"type": "status", "status": "training_started"})
                await _send_status()
            elif cmd == "pause":
                training_manager.pause()
                await _send_status()
            elif cmd == "resume":
                training_manager.resume()
                await _send_status()
            elif cmd == "stop":
                training_manager.stop()
                await _send_status()
            elif cmd == "step_batch":
                await ws.send_json(training_manager.step_batch())
            elif cmd == "step_epoch":
                await ws.send_json(training_manager.step_epoch())
            elif cmd in {"get_status", "status"}:
                await _send_status()
            else:
                await ws.send_json({"type": "error", "message": f"Unknown command: {cmd}"})
    except WebSocketDisconnect:
        return
