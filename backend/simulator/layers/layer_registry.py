from __future__ import annotations

from typing import Dict, Type

from .batchnorm import BatchNormLayer
from .conv2d import Conv2DLayer
from .dense import DenseLayer
from .flatten import FlattenLayer
from .attention import AttentionLayer
from .embedding import EmbeddingLayer
from .gru import GRULayer
from .lstm import LSTMLayer
from .pooling import AvgPool2DLayer, MaxPool2DLayer
from .residual import ResidualLayer
from .rnn import RNNLayer


LAYER_REGISTRY: Dict[str, Type] = {
    "dense": DenseLayer,
    "conv2d": Conv2DLayer,
    "maxpool2d": MaxPool2DLayer,
    "avgpool2d": AvgPool2DLayer,
    "flatten": FlattenLayer,
    "batchnorm": BatchNormLayer,
    "rnn": RNNLayer,
    "lstm": LSTMLayer,
    "gru": GRULayer,
    "embedding": EmbeddingLayer,
    "attention": AttentionLayer,
    "residual": ResidualLayer,
}
