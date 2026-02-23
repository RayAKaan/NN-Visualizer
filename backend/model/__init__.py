from .ann_model import build_ann_model
from .cnn_model import build_cnn_model
from .rnn_model import build_rnn_model

MODEL_BUILDERS = {
    "ann": build_ann_model,
    "cnn": build_cnn_model,
    "rnn": build_rnn_model,
}
