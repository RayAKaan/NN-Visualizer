from __future__ import annotations

from typing import List, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_utils import (
    load_model,
    build_activation_model,
    extract_weights,
    normalize_pixels,
)

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
# Schemas
# ============================================================

class PixelInput(BaseModel):
    pixels: List[float]

# ============================================================
# Helpers
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
    hidden1: np.ndarray,
    hidden2: np.ndarray,
    probabilities: np.ndarray,
    w2o: np.ndarray,
) -> Dict:
    # ---- hard guarantees ----
    assert hidden1.shape == (128,)
    assert hidden2.shape == (64,)
    assert probabilities.shape == (10,)
    assert w2o.shape == (64, 10)

    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    # ---- contribution math ----
    output_weights = w2o[:, prediction]
    contributions = hidden2 * output_weights

    top_idx = np.argsort(np.abs(contributions))[-5:][::-1]

    top_neurons = [
        {
            "neuron": int(i),
            "activation": float(hidden2[i]),
            "weight": float(output_weights[i]),
            "contribution": float(contributions[i]),
        }
        for i in top_idx
    ]

    quadrant_intensity = summarize_quadrants(pixels)

    positive_support = np.maximum(0.0, hidden2[:, None] * w2o)
    support_scores = np.sum(positive_support, axis=0)

    top_digits = np.argsort(probabilities)[-3:][::-1]

    competing = [
        {
            "digit": int(d),
            "probability": float(probabilities[d]),
            "positive_support": float(support_scores[d]),
        }
        for d in top_digits
        if d != prediction
    ]

    margin = (
        float(confidence - probabilities[top_digits[1]])
        if len(top_digits) > 1
        else confidence
    )

    notes: List[str] = []
    if confidence < 0.6:
        notes.append("Low confidence prediction.")
    if margin < 0.15:
        notes.append("Top predictions are very close.")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "margin": margin,
        "top_hidden2_neurons": top_neurons,
        "quadrant_intensity": quadrant_intensity,
        "competing_digits": competing,
        "uncertainty_notes": notes,
        "hidden1_summary": {
            "mean_activation": float(np.mean(hidden1)),
            "max_activation": float(np.max(hidden1)),
        },
    }


def build_neural_state(
    pixels: np.ndarray,
    hidden1: np.ndarray,
    hidden2: np.ndarray,
    probabilities: np.ndarray,
    w12: np.ndarray,
    w2o: np.ndarray,
) -> Dict:
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])

    edges_h1_h2 = [
        {
            "from": i,
            "to": j,
            "strength": float(hidden1[i] * w12[i, j]),
        }
        for i in range(128)
        for j in range(64)
        if abs(hidden1[i] * w12[i, j]) > 0.05
    ]

    edges_h2_out = [
        {
            "from": j,
            "to": prediction,
            "strength": float(hidden2[j] * w2o[j, prediction]),
        }
        for j in range(64)
        if abs(hidden2[j] * w2o[j, prediction]) > 0.05
    ]

    return {
        "input": pixels.tolist(),
        "layers": {
            "hidden1": hidden1.tolist(),
            "hidden2": hidden2.tolist(),
            "output": probabilities.tolist(),
        },
        "prediction": prediction,
        "confidence": confidence,
        "edges": {
            "hidden1_hidden2": edges_h1_h2,
            "hidden2_output": edges_h2_out,
        },
    }

# ============================================================
# Model loading
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
# Routes
# ============================================================

@app.post("/predict")
async def predict(data: PixelInput) -> Dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    x = normalize_pixels(data.pixels)

    # Guaranteed order: [hidden1, hidden2, output]
    hidden1, hidden2, probabilities = ACTIVATION_MODEL.predict(x, verbose=0)

    _, _, w2o = extract_weights(MODEL)

    explanation = build_explanation(
        pixels=x[0],
        hidden1=hidden1[0],
        hidden2=hidden2[0],
        probabilities=probabilities[0],
        w2o=w2o,
    )

    return {
        "prediction": int(np.argmax(probabilities[0])),
        "probabilities": probabilities[0].tolist(),
        "layers": {
            "hidden1": hidden1[0].tolist(),
            "hidden2": hidden2[0].tolist(),
        },
        "explanation": explanation,
    }


@app.post("/state")
async def neural_state(data: PixelInput) -> Dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    x = normalize_pixels(data.pixels)

    hidden1, hidden2, probabilities = ACTIVATION_MODEL.predict(x, verbose=0)
    _, w12, w2o = extract_weights(MODEL)

    return build_neural_state(
        pixels=x[0],
        hidden1=hidden1[0],
        hidden2=hidden2[0],
        probabilities=probabilities[0],
        w12=w12,
        w2o=w2o,
    )


@app.get("/weights")
async def weights() -> Dict:
    if MODEL is None:
        raise HTTPException(status_code=500, detail=MODEL_ERROR)

    _, w12, w2o = extract_weights(MODEL)
    return {
        "hidden1_hidden2": w12.tolist(),
        "hidden2_output": w2o.tolist(),
    }