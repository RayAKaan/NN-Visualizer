import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from training.manager import training_manager

router = APIRouter()


@router.websocket("/train")
async def train_ws(ws: WebSocket):
    await ws.accept()

    def emit(payload):
        asyncio.create_task(ws.send_json(payload))

    try:
        while True:
            data = json.loads(await ws.receive_text())
            cmd = data.get("command")
            if cmd == "configure":
                await ws.send_json(training_manager.configure(data.get("config", {})))
            elif cmd == "start":
                training_manager.start(emit)
            elif cmd == "pause":
                training_manager.pause()
            elif cmd == "resume":
                training_manager.resume()
            elif cmd == "stop":
                training_manager.stop()
            elif cmd == "step_batch":
                await ws.send_json(training_manager.step_batch())
            elif cmd == "step_epoch":
                await ws.send_json(training_manager.step_epoch())
            elif cmd == "get_status":
                await ws.send_json(training_manager.get_status())
            else:
                await ws.send_json({"type": "error", "message": f"Unknown command: {cmd}"})
    except WebSocketDisconnect:
        return
