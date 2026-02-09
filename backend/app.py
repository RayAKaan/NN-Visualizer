from __future__ import annotations

from typing import List

import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import build_activation_model, extract_weights, load_model, normalize_pixels
from training.manager import TrainingConfig, TrainingManager

app = FastAPI(title="NN Visualizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PixelInput(BaseModel):
    pixels: List[float]


def summarize_quadrants(pixels: np.ndarray) -> dict:
    grid = pixels.reshape(28, 28)
    quadrants = {
        "upper_left": float(np.mean(grid[:14, :14])),
        "upper_right": float(np.mean(grid[:14, 14:])),
        "lower_left": float(np.mean(grid[14:, :14])),
        "lower_right": float(np.mean(grid[14:, 14:])),
    }
    return quadrants


def build_explanation(
    pixels: np.ndarray,
    hidden1: np.ndarray,
    hidden2: np.ndarray,
    probabilities: np.ndarray,
    weights_hidden2_output: np.ndarray,
) -> dict:
    prediction = int(np.argmax(probabilities))
    contributions = hidden2 * weights_hidden2_output[:, prediction]
    top_indices = np.argsort(contributions)[-5:][::-1]
    top_neurons = [
        {
            "neuron": int(index),
            "activation": float(hidden2[index]),
            "weight": float(weights_hidden2_output[index, prediction]),
            "contribution": float(contributions[index]),
        }
        for index in top_indices
    ]

    quadrant_intensity = summarize_quadrants(pixels)

    positive_support = np.maximum(0, hidden2[:, None] * weights_hidden2_output)
    support_scores = np.sum(positive_support, axis=0)
    top_digits = np.argsort(probabilities)[-3:][::-1]
    competing = [
        {
            "digit": int(digit),
            "probability": float(probabilities[digit]),
            "positive_support": float(support_scores[digit]),
        }
        for digit in top_digits
        if digit != prediction
    ]

    confidence = float(probabilities[prediction])
    margin = float(confidence - probabilities[top_digits[1]]) if len(top_digits) > 1 else confidence
    uncertainty_notes: List[str] = []
    if confidence < 0.6:
        uncertainty_notes.append("Low top probability; prediction is uncertain.")
    if margin < 0.15:
        uncertainty_notes.append("Top two probabilities are close; digits are competing.")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "top_hidden2_neurons": top_neurons,
        "quadrant_intensity": quadrant_intensity,
        "competing_digits": competing,
        "uncertainty_notes": uncertainty_notes,
        "hidden1_summary": {
            "mean_activation": float(np.mean(hidden1)),
            "max_activation": float(np.max(hidden1)),
        },
    }


def build_edges(
    weights: np.ndarray, source_activations: np.ndarray, max_edges: int
) -> List[dict]:
    contributions = weights * source_activations[:, None]
    flat = [
        (i, j, contributions[i, j])
        for i in range(contributions.shape[0])
        for j in range(contributions.shape[1])
    ]
    flat.sort(key=lambda item: abs(item[2]), reverse=True)
    top = flat[:max_edges]
    return [
        {"from": int(i), "to": int(j), "strength": float(strength)} for i, j, strength in top
    ]


try:
    MODEL = load_model()
    ACTIVATION_MODEL = build_activation_model(MODEL)
except FileNotFoundError as exc:
    MODEL = None
    ACTIVATION_MODEL = None
    MODEL_ERROR = str(exc)
else:
    MODEL_ERROR = None

TRAINING_MANAGER = TrainingManager()


async def send_training_message(message: dict) -> None:
    await websocket_manager.broadcast(message)


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


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.websocket("/ws/training")
async def training_ws(websocket: WebSocket) -> None:
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            command = data.get("command")
            if command == "configure":
                config = TrainingConfig(
                    learning_rate=float(data.get("learning_rate", 0.001)),
                    batch_size=int(data.get("batch_size", 64)),
                    epochs=int(data.get("epochs", 5)),
                    optimizer=str(data.get("optimizer", "adam")).lower(),
                    weight_decay=float(data.get("weight_decay", 0.0)),
                    activation=str(data.get("activation", "relu")).lower(),
                    initializer=str(data.get("initializer", "glorot_uniform")),
                    dropout=float(data.get("dropout", 0.0)),
                )
                TRAINING_MANAGER.configure(config)
                await websocket.send_json({"type": "configured"})
            elif command == "start":
                loop = asyncio.get_running_loop()
                TRAINING_MANAGER.start(send_training_message, loop)
            elif command == "pause":
                TRAINING_MANAGER.pause()
            elif command == "resume":
                TRAINING_MANAGER.resume()
            elif command == "stop":
                TRAINING_MANAGER.stop()
            elif command == "step_batch":
                TRAINING_MANAGER.step_batch()
            elif command == "step_epoch":
                TRAINING_MANAGER.step_epoch()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


@app.post("/predict")
async def predict(data: PixelInput) -> dict:
    if MODEL is None or ACTIVATION_MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    x = normalize_pixels(data.pixels)
    activations = ACTIVATION_MODEL.predict(x)
    probabilities = activations[-1][0]
    prediction = int(np.argmax(probabilities))
    _, _, weights_hidden2_output = extract_weights(MODEL)
    explanation = build_explanation(
        x[0],
        activations[0][0],
        activations[1][0],
        probabilities,
        weights_hidden2_output,
    )

    return {
        "prediction": prediction,
        "probabilities": probabilities.tolist(),
        "layers": {
            "hidden1": activations[0][0].tolist(),
            "hidden2": activations[1][0].tolist(),
        },
        "explanation": explanation,
    }


@app.post("/state")
async def state(data: PixelInput) -> dict:
    if MODEL is None or ACTIVATION_MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    x = normalize_pixels(data.pixels)
    activations = ACTIVATION_MODEL.predict(x)
    probabilities = activations[-1][0]
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    _, weights_hidden_hidden, weights_hidden_output = extract_weights(MODEL)

    edges_hidden1_hidden2 = build_edges(weights_hidden_hidden, activations[0][0], 240)
    edges_hidden2_output = build_edges(weights_hidden_output, activations[1][0], 160)

    return {
        "input": x[0].tolist(),
        "layers": {
            "hidden1": activations[0][0].tolist(),
            "hidden2": activations[1][0].tolist(),
            "output": probabilities.tolist(),
        },
        "prediction": prediction,
        "confidence": confidence,
        "edges": {
            "hidden1_hidden2": edges_hidden1_hidden2,
            "hidden2_output": edges_hidden2_output,
        },
    }


@app.get("/weights")
async def weights() -> dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)
    _, hidden_hidden, hidden_output = extract_weights(MODEL)
    return {
        "hidden1_hidden2": hidden_hidden.tolist(),
        "hidden2_output": hidden_output.tolist(),
    }
