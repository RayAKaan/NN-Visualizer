import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_MODEL_PATH = os.path.join(BASE_DIR, "nn_visual_ann.h5")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "nn_visual_cnn.h5")
RNN_MODEL_PATH = os.path.join(BASE_DIR, "nn_visual_rnn.h5")
MODELS_DIR = os.path.join(BASE_DIR, "models_store")
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
