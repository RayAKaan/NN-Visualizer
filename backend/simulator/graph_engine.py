from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import hashlib

import numpy as np

from .activations import get_activation
from .layers.batchnorm import BatchNormLayer
from .layers.conv2d import Conv2DConfig, Conv2DLayer
from .layers.dense import DenseLayer
from .layers.flatten import FlattenLayer
from .layers.input_layer import InputLayer
from .layers.attention import AttentionLayer
from .layers.embedding import EmbeddingLayer
from .layers.gru import GRULayer
from .layers.lstm import LSTMLayer
from .layers.pooling import AvgPool2DLayer, MaxPool2DLayer, PoolConfig
from .layers.rnn import RNNLayer
from .layers.residual import ResidualLayer
from .layers import LayerConfig, validate_layers


def _init_weights(shape: Tuple[int, int], init: str | None) -> np.ndarray:
    init_name = (init or "xavier").lower()
    fan_out, fan_in = shape
    if init_name in {"xavier", "glorot"}:
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif init_name == "he":
        std = np.sqrt(2.0 / fan_in)
    elif init_name == "lecun":
        std = np.sqrt(1.0 / fan_in)
    elif init_name == "zeros":
        return np.zeros(shape, dtype=np.float32)
    else:
        std = 0.01
    return np.random.normal(0.0, std, size=shape).astype(np.float32)


def _init_bias(shape: Tuple[int], init: str | None) -> np.ndarray:
    init_name = (init or "zeros").lower()
    if init_name == "random":
        return np.random.normal(0.0, 0.01, size=shape).astype(np.float32)
    return np.zeros(shape, dtype=np.float32)


@dataclass
class NetworkGraph:
    layers: List[LayerConfig]
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    layer_instances: List[object] = field(default_factory=list)
    param_layer_indices: List[int] = field(default_factory=list)
    param_index_by_layer: Dict[int, int] = field(default_factory=dict)
    pre_activations: List[np.ndarray] = field(default_factory=list)
    activations: List[np.ndarray] = field(default_factory=list)
    total_params: int = 0
    flops_per_sample: int = 0
    architecture_hash: str = ""
    input_shape: Tuple[int, ...] | None = None

    def forward(self, input_vec: np.ndarray) -> np.ndarray:
        if self.layer_instances:
            self.pre_activations = []
            self.activations = []
            x = input_vec.astype(np.float32)
            for layer in self.layer_instances:
                x = layer.forward(x)
                if hasattr(layer, "Z") and getattr(layer, "Z") is not None:
                    self.pre_activations.append(layer.Z)
                self.activations.append(x)
            return x
        self.pre_activations = []
        self.activations = [input_vec.astype(np.float32)]
        for idx in range(1, len(self.layers)):
            w = self.weights[idx - 1]
            b = self.biases[idx - 1]
            z = w @ self.activations[-1] + b
            self.pre_activations.append(z)
            act_name = (self.layers[idx].activation or "linear").lower()
            a = get_activation(act_name).forward(z)
            self.activations.append(a)
        return self.activations[-1]


def build_graph(layers: List[LayerConfig]) -> NetworkGraph:
    validation = validate_layers(layers)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    layer_instances: List[object] = []
    param_layer_indices: List[int] = []
    param_index_by_layer: Dict[int, int] = {}

    if layers and layers[0].input_shape:
        input_shape = tuple(int(v) for v in layers[0].input_shape)
    elif layers and layers[0].sequence_length and layers[0].neurons:
        input_shape = (layers[0].sequence_length, layers[0].neurons)
    elif layers:
        input_shape = (layers[0].neurons,)
    else:
        input_shape = (0,)

    current_shape = input_shape
    layer_instances.append(InputLayer(input_shape))

    for idx in range(1, len(layers)):
        layer = layers[idx]
        ltype = layer.layer_type
        if ltype in {"dense", "output"}:
            in_dim = int(current_shape[0])
            out_dim = layer.neurons
            dense = DenseLayer(in_dim, out_dim, layer.activation, layer.init)
            layer_instances.append(dense)
            current_shape = (out_dim,)
        elif ltype == "batchnorm":
            features = current_shape[0]
            bn = BatchNormLayer(features)
            layer_instances.append(bn)
        elif ltype == "conv2d":
            c_in, h_in, w_in = current_shape
            k = layer.kernel_size or 3
            stride = layer.stride or 1
            padding = layer.padding or "valid"
            c_out = layer.filters or layer.neurons
            conv = Conv2DLayer(
                Conv2DConfig(
                    c_in=c_in,
                    c_out=c_out,
                    kernel_size=(k, k),
                    stride=stride,
                    padding=padding,
                    activation=layer.activation or "relu",
                    init=layer.init or "he",
                )
            )
            layer_instances.append(conv)
            current_shape = conv.output_shape(current_shape)
        elif ltype in {"maxpool2d", "avgpool2d"}:
            pool_size = layer.pool_size or layer.kernel_size or 2
            stride = layer.pool_stride or layer.stride or pool_size
            cfg = PoolConfig(pool_size=pool_size, stride=stride)
            pool = MaxPool2DLayer(cfg) if ltype == "maxpool2d" else AvgPool2DLayer(cfg)
            layer_instances.append(pool)
            c_in, h_in, w_in = current_shape
            h_out = (h_in - pool_size) // stride + 1
            w_out = (w_in - pool_size) // stride + 1
            current_shape = (c_in, h_out, w_out)
        elif ltype == "flatten":
            layer_instances.append(FlattenLayer())
            size = 1
            for dim in current_shape:
                size *= dim
            current_shape = (size,)
        elif ltype == "rnn":
            t_len, d_in = current_shape
            hidden = layer.hidden_size or layer.neurons
            return_seq = bool(layer.return_sequences)
            rnn = RNNLayer(d_in, hidden, return_sequences=return_seq)
            layer_instances.append(rnn)
            current_shape = (t_len, hidden) if return_seq else (hidden,)
        elif ltype == "lstm":
            t_len, d_in = current_shape
            hidden = layer.hidden_size or layer.neurons
            return_seq = bool(layer.return_sequences)
            lstm = LSTMLayer(d_in, hidden, return_sequences=return_seq)
            layer_instances.append(lstm)
            current_shape = (t_len, hidden) if return_seq else (hidden,)
        elif ltype == "gru":
            t_len, d_in = current_shape
            hidden = layer.hidden_size or layer.neurons
            return_seq = bool(layer.return_sequences)
            gru = GRULayer(d_in, hidden, return_sequences=return_seq)
            layer_instances.append(gru)
            current_shape = (t_len, hidden) if return_seq else (hidden,)
        elif ltype == "embedding":
            t_len = current_shape[0]
            vocab = layer.vocab_size or 50
            emb = layer.embedding_dim or layer.neurons
            emb_layer = EmbeddingLayer(vocab, emb)
            layer_instances.append(emb_layer)
            current_shape = (t_len, emb)
        elif ltype == "attention":
            t_len, d_model = current_shape
            heads = layer.num_heads or 1
            attn = AttentionLayer(d_model, num_heads=heads)
            layer_instances.append(attn)
            current_shape = (t_len, d_model)
        elif ltype == "residual":
            layer_instances.append(ResidualLayer())
        else:
            raise ValueError(f"Unsupported layer type '{ltype}'")

    for idx, layer in enumerate(layer_instances):
        if getattr(layer, "has_params", False):
            w, b = layer.params()
            if w is None or b is None:
                continue
            param_index_by_layer[idx] = len(weights)
            param_layer_indices.append(idx)
            weights.append(w)
            biases.append(b)

    arch_sig = ",".join(str(l.neurons) for l in layers)
    arch_hash = hashlib.sha1(arch_sig.encode("utf-8")).hexdigest()[:12]
    return NetworkGraph(
        layers=layers,
        weights=weights,
        biases=biases,
        layer_instances=layer_instances,
        param_layer_indices=param_layer_indices,
        param_index_by_layer=param_index_by_layer,
        total_params=validation.total_params,
        flops_per_sample=validation.flops_per_sample,
        architecture_hash=arch_hash,
        input_shape=input_shape,
    )
