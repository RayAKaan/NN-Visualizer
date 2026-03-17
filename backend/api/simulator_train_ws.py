from __future__ import annotations

import asyncio
import json
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from simulator.dataset_manager import dataset_manager
from simulator.session_manager import session_manager
from simulator.training_engine import TrainingConfig, training_sessions
from simulator.debugger import diagnose

router = APIRouter()


@router.websocket("/ws/simulator/train")
async def simulator_train_ws(ws: WebSocket):
    await ws.accept()
    training_task: asyncio.Task | None = None
    stop_flag = False
    current_graph_id: str | None = None

    async def run_training(graph_id: str, dataset_id: str, config: TrainingConfig):
        nonlocal stop_flag
        graph = session_manager.get_graph(graph_id)
        session = training_sessions.reset(graph_id, graph, config)
        data = dataset_manager.get(dataset_id)
        train = [(np.array(p["x"], dtype=np.float32), np.array(p["y"], dtype=np.float32)) for p in data["train"]]
        test = [(np.array(p["x"], dtype=np.float32), np.array(p["y"], dtype=np.float32)) for p in data["test"]]

        await ws.send_json({"type": "status", "status": "training", "message": "Training started"})

        for epoch in range(config.epochs):
            if stop_flag:
                break
            while session.is_paused:
                await asyncio.sleep(0.2)
            session.epoch = epoch

            def batch_cb(batch_index, total_batches, loss, acc, grad_norm):
                asyncio.create_task(
                    ws.send_json(
                        {
                            "type": "batch",
                            "epoch": epoch,
                            "batch": batch_index + 1,
                            "total_batches": total_batches,
                            "loss": loss,
                            "accuracy": acc,
                            "gradient_norm": grad_norm,
                        }
                    )
                )

            metrics = session.train_epoch(train, test, graph_id, batch_callback=batch_cb)

            await ws.send_json(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "metrics": {
                        "train_loss": metrics.train_loss,
                        "test_loss": metrics.test_loss,
                        "train_accuracy": metrics.train_accuracy,
                        "test_accuracy": metrics.test_accuracy,
                        "learning_rate": metrics.learning_rate,
                        "gradient_norms": metrics.gradient_norms,
                        "weight_norms": metrics.weight_norms,
                        "weight_deltas": metrics.weight_deltas,
                        "dead_neurons": metrics.dead_neurons,
                        "epoch_duration_ms": metrics.epoch_duration_ms,
                    },
                }
            )

            diag = diagnose(metrics.gradient_norms, [m.train_loss for m in session.history], metrics.train_loss, metrics.test_loss, metrics.dead_neurons, [l.neurons for l in graph.layers[1:]])
            for issue in diag["issues"]:
                await ws.send_json(
                    {
                        "type": "warning",
                        "severity": issue.get("severity", "warning"),
                        "code": issue.get("code"),
                        "message": issue.get("message"),
                        "layer_index": issue.get("layer_index"),
                        "epoch": epoch,
                        "suggestion": issue.get("suggestion"),
                    }
                )

        await ws.send_json(
            {
                "type": "complete",
                "total_epochs": config.epochs,
                "final_train_loss": session.history[-1].train_loss if session.history else 0,
                "final_test_loss": session.history[-1].test_loss if session.history else 0,
                "final_train_accuracy": session.history[-1].train_accuracy if session.history else 0,
                "final_test_accuracy": session.history[-1].test_accuracy if session.history else 0,
                "total_snapshots": len(session.history),
            }
        )

    try:
        while True:
            data = json.loads(await ws.receive_text())
            action = data.get("action")
            if action == "start":
                graph_id = data.get("graph_id")
                dataset_id = data.get("dataset_id")
                current_graph_id = graph_id
                cfg = data.get("config", {})
                config = TrainingConfig(**cfg)
                stop_flag = False
                training_task = asyncio.create_task(run_training(graph_id, dataset_id, config))
            elif action == "pause":
                if training_task and current_graph_id:
                    session = training_sessions.get(current_graph_id)
                    session.is_paused = True
                    await ws.send_json({"type": "status", "status": "paused"})
            elif action == "resume":
                if training_task and current_graph_id:
                    session = training_sessions.get(current_graph_id)
                    session.is_paused = False
                    await ws.send_json({"type": "status", "status": "training"})
            elif action == "stop":
                stop_flag = True
                await ws.send_json({"type": "status", "status": "stopped"})
            elif action == "update_config":
                graph_id = data.get("graph_id") or current_graph_id
                updates = data.get("updates", {})
                session = training_sessions.get(graph_id)
                for k, v in updates.items():
                    if hasattr(session.config, k):
                        setattr(session.config, k, v)
                if "optimizer" in updates:
                    session.optimizer_state = None
            else:
                await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        if training_task:
            training_task.cancel()
        return
