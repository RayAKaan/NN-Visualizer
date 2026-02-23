import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from training.manager import training_manager

router = APIRouter()


@router.websocket("/train")
async def train_websocket(ws: WebSocket):
    await ws.accept()

    message_queue: asyncio.Queue = asyncio.Queue()

    def on_training_message(msg: dict):
        """Called from training thread. Puts message into async queue."""
        try:
            message_queue.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    training_manager.ws_callback = on_training_message

    async def send_loop():
        while True:
            try:
                msg = await asyncio.wait_for(message_queue.get(), timeout=0.05)
                await ws.send_json(msg)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

    sender_task = asyncio.create_task(send_loop())

    try:
        while True:
            raw = await ws.receive_text()
            command = json.loads(raw)
            cmd = command.get("command", "")
            response = {"type": "command_response", "command": cmd}

            if cmd == "configure":
                result = training_manager.configure(command.get("config", {}))
                response.update(result)

            elif cmd == "start":
                result = training_manager.start()
                response.update(result)

            elif cmd == "pause":
                result = training_manager.pause()
                response.update(result)

            elif cmd == "resume":
                result = training_manager.resume()
                response.update(result)

            elif cmd == "stop":
                result = training_manager.stop()
                response.update(result)

            elif cmd == "step_batch":
                result = training_manager.step_batch()
                response.update(result)

            elif cmd == "step_epoch":
                result = training_manager.step_epoch()
                response.update(result)

            elif cmd == "get_status":
                result = training_manager.get_status()
                response.update(result)

            else:
                response["error"] = f"Unknown command: {cmd}"

            await ws.send_json(response)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        sender_task.cancel()
        training_manager.ws_callback = None
