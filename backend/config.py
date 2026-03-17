import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_store")
os.makedirs(MODELS_DIR, exist_ok=True)

def _resolve_model_path(filename: str) -> str:
    preferred = os.path.join(MODELS_DIR, filename)
    if os.path.exists(preferred):
        return preferred
    # Backward-compatible fallback for older repos that keep models in backend/.
    legacy = os.path.join(BASE_DIR, filename)
    if os.path.exists(legacy):
        return legacy
    return preferred


ANN_MODEL_PATH = _resolve_model_path("nn_visual_ann.h5")
CNN_MODEL_PATH = _resolve_model_path("nn_visual_cnn.h5")
RNN_MODEL_PATH = _resolve_model_path("nn_visual_rnn.h5")
CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]
SUPPORTED_MODEL_TYPES = ["ann", "cnn", "rnn"]
DEFAULT_MODEL_TYPE = "ann"

MODEL_PATHS = {
    "ann": ANN_MODEL_PATH,
    "cnn": CNN_MODEL_PATH,
    "rnn": RNN_MODEL_PATH,
}

MODEL_INPUT_SHAPES = {
    "ann": (784,),
    "cnn": (28, 28, 1),
    "rnn": (28, 28),
}
