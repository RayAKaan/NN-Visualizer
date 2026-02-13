from __future__ import annotations

import asyncio
from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import (
    load_model,
    build_activation_model,
    extract_weights,
    normalize_pixels,
)

from training.manager import TrainingManager, TrainingConfig

# ============================================================
# App setup
# ============================================================

app = FastAPI(title="NN Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global objects
# ============================================================

training_manager = TrainingManager()

# ============================================================
# Schemas
# ============================================================

class PixelInput(BaseModel):
    pixels: List[float]


# ============================================================
# Explanation helpers
# ============================================================

def summarize_quadrants(pixels: np.ndarray) -> Dict[str, float]:
    grid = pixels.reshape(28, 28)
    return {
        "upper_left": float(np.mean(grid[:14, :14])),
        "upper_right": float(np.mean(grid[:14, 14:])),
        "lower_left": float(np.mean(grid[14:, :14])),
        "lower_right": float(np.mean(grid[14:, 14:])),
    }


def build_explanation(
    pixels: np.ndarray,
    hidden2: np.ndarray,
    probs: np.ndarray,
    w2o: np.ndarray,
) -> Dict:
    """
    hidden2: (64,)
    w2o: (64, 10)
    probs: (10,)
    """
    prediction = int(np.argmax(probs))
    confidence = float(probs[prediction])

    contrib = hidden2 * w2o[:, prediction]
    top_idx = np.argsort(np.abs(contrib))[-5:][::-1]

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_hidden2_neurons": [
            {
                "neuron": int(i),
                "activation": float(hidden2[i]),
                "contribution": float(contrib[i]),
            }
            for i in top_idx
        ],
        "quadrant_intensity": summarize_quadrants(pixels),
    }


# ============================================================
# Load inference model
# ============================================================

try:
    MODEL = load_model()
    ACTIVATION_MODEL = build_activation_model(MODEL)
except FileNotFoundError as exc:
    MODEL = None
    ACTIVATION_MODEL = None
    MODEL_ERROR = str(exc)
else:
    MODEL_ERROR = None


# ============================================================
# REST: Inference
# ============================================================

@app.post("/predict")
async def predict(data: PixelInput) -> Dict:
    if MODEL is None or ACTIVATION_MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    x = normalize_pixels(data.pixels)

    # activation model outputs:
    # [hidden1, hidden2, output]
    hidden1, hidden2, probs = ACTIVATION_MODEL.predict(x, verbose=0)

    _, _, w2o = extract_weights(MODEL)

    explanation = build_explanation(
        pixels=x[0].reshape(28, 28),
        hidden2=hidden2[0],
        probs=probs[0],
        w2o=w2o,
    )

    return {
        "prediction": int(np.argmax(probs[0])),
        "probabilities": probs[0].tolist(),
        "layers": {
            "hidden1": hidden1[0].tolist(),
            "hidden2": hidden2[0].tolist(),
        },
        "explanation": explanation,
    }


@app.get("/weights")
async def weights() -> Dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    _, w12, w2o = extract_weights(MODEL)

    return {
        "hidden1_hidden2": w12.tolist(),  # (128, 64)
        "hidden2_output": w2o.tolist(),   # (64, 10)
    }


# ============================================================
# WebSocket: Training Control + Streaming
# ============================================================

@app.websocket("/train")
async def train_socket(ws: WebSocket):
    """
    WebSocket protocol:

    Incoming:
      { command: "configure", config: {...} }
      { command: "start" }
      { command: "pause" }
      { command: "resume" }
      { command: "stop" }
      { command: "step_batch" }
      { command: "step_epoch" }

    Outgoing:
      batch / epoch / weights / stopped
    """

    await ws.accept()
    loop = asyncio.get_event_loop()

    async def emit(message: dict):
        await ws.send_json(message)

    try:
        while True:
            msg = await ws.receive_json()
            command = msg.get("command")

            if command == "configure":
                training_manager.configure(
                    TrainingConfig(**msg["config"])
                )

            elif command == "start":
                training_manager.start(emit=emit, loop=loop)

            elif command == "pause":
                training_manager.pause()

            elif command == "resume":
                training_manager.resume()

            elif command == "stop":
                training_manager.stop()

            elif command == "step_batch":
                training_manager.step_batch()

            elif command == "step_epoch":
                training_manager.step_epoch()

    except WebSocketDisconnect:
        training_manager.stop()
