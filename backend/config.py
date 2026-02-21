from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nn_visual.h5"
CNN_MODEL_PATH = BASE_DIR / "nn_visual_cnn.keras"
CNN_MODEL_PATH_H5 = BASE_DIR / "nn_visual_cnn.h5"
MODELS_DIR = BASE_DIR / "models_store"
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
SUPPORTED_MODEL_TYPES = ["ann", "cnn"]
DEFAULT_MODEL_TYPE = "ann"
SAMPLES_PATH = BASE_DIR / "sample_digits.json"
