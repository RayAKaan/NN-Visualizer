from __future__ import annotations

from typing import List

from ..layers import LayerConfig


def _act(name: str | None) -> str:
    if not name or name == "linear":
        return ""
    return name


def _pytorch_activation(name: str | None) -> str:
    name = (name or "").lower()
    if name in {"relu", "leaky_relu"}:
        return "nn.ReLU()" if name == "relu" else "nn.LeakyReLU()"
    if name == "sigmoid":
        return "nn.Sigmoid()"
    if name == "tanh":
        return "nn.Tanh()"
    return "None"


def _keras_activation(name: str | None) -> str:
    name = (name or "").lower()
    if name in {"relu", "leaky_relu", "sigmoid", "tanh", "softmax", "linear"}:
        return name
    return "linear"


def generate_pytorch(layers: List[LayerConfig]) -> str:
    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "class CustomModel(nn.Module):",
        "    def __init__(self):",
        "        super().__init__()",
    ]
    layer_defs = []
    forward_lines = ["    def forward(self, x):"]

    for idx, layer in enumerate(layers[1:], start=1):
        ltype = layer.layer_type
        name = f"layer{idx}"
        if ltype in {"dense", "output"}:
            in_dim = layer.neurons
            if idx > 1:
                in_dim = layers[idx - 1].neurons
            layer_defs.append(f"        self.{name} = nn.Linear({in_dim}, {layer.neurons})")
            forward_lines.append(f"        x = self.{name}(x)")
            act = _pytorch_activation(layer.activation)
            if act != "None":
                forward_lines.append(f"        x = {act}(x)")
        elif ltype == "conv2d":
            c_out = layer.filters or layer.neurons
            k = layer.kernel_size or 3
            stride = layer.stride or 1
            pad = 1 if (layer.padding or "valid") == "same" else 0
            c_in = layers[idx - 1].neurons or 1
            layer_defs.append(f"        self.{name} = nn.Conv2d({c_in}, {c_out}, {k}, stride={stride}, padding={pad})")
            forward_lines.append(f"        x = self.{name}(x)")
            act = _pytorch_activation(layer.activation)
            if act != "None":
                forward_lines.append(f"        x = {act}(x)")
        elif ltype in {"maxpool2d", "avgpool2d"}:
            k = layer.pool_size or layer.kernel_size or 2
            stride = layer.pool_stride or layer.stride or k
            cls = "nn.MaxPool2d" if ltype == "maxpool2d" else "nn.AvgPool2d"
            layer_defs.append(f"        self.{name} = {cls}({k}, stride={stride})")
            forward_lines.append(f"        x = self.{name}(x)")
        elif ltype == "flatten":
            layer_defs.append(f"        self.{name} = nn.Flatten()")
            forward_lines.append(f"        x = self.{name}(x)")
        elif ltype == "batchnorm":
            features = layers[idx - 1].neurons
            layer_defs.append(f"        self.{name} = nn.BatchNorm1d({features})")
            forward_lines.append(f"        x = self.{name}(x)")
        elif ltype in {"rnn", "lstm", "gru"}:
            hidden = layer.hidden_size or layer.neurons
            d_in = layers[idx - 1].neurons
            cls = "nn.RNN" if ltype == "rnn" else "nn.LSTM" if ltype == "lstm" else "nn.GRU"
            layer_defs.append(f"        self.{name} = {cls}({d_in}, {hidden}, batch_first=True)")
            forward_lines.append(f"        x, _ = self.{name}(x)")
            if not layer.return_sequences:
                forward_lines.append("        x = x[:, -1, :]")
        elif ltype == "embedding":
            vocab = layer.vocab_size or 50
            emb = layer.embedding_dim or layer.neurons
            layer_defs.append(f"        self.{name} = nn.Embedding({vocab}, {emb})")
            forward_lines.append(f"        x = self.{name}(x)")
        elif ltype == "attention":
            heads = layer.num_heads or 1
            d_model = layers[idx - 1].neurons
            layer_defs.append(f"        self.{name} = nn.MultiheadAttention({d_model}, {heads}, batch_first=True)")
            forward_lines.append(f"        x, _ = self.{name}(x, x, x)")
        elif ltype == "residual":
            forward_lines.append("        x = x + x")
        else:
            forward_lines.append("        # Unsupported layer")

    lines.extend(layer_defs)
    lines.append("")
    forward_lines.append("        return x")
    lines.extend(forward_lines)
    return "\n".join(lines)


def generate_keras(layers: List[LayerConfig]) -> str:
    lines = [
        "from tensorflow import keras",
        "",
        "model = keras.Sequential(["]

    for idx, layer in enumerate(layers[1:], start=1):
        ltype = layer.layer_type
        if ltype in {"dense", "output"}:
            act = _keras_activation(layer.activation)
            lines.append(f"    keras.layers.Dense({layer.neurons}, activation='{act}'),")
        elif ltype == "conv2d":
            c_out = layer.filters or layer.neurons
            k = layer.kernel_size or 3
            pad = "same" if (layer.padding or "valid") == "same" else "valid"
            act = _keras_activation(layer.activation)
            lines.append(f"    keras.layers.Conv2D({c_out}, {k}, padding='{pad}', activation='{act}'),")
        elif ltype in {"maxpool2d", "avgpool2d"}:
            k = layer.pool_size or layer.kernel_size or 2
            cls = "MaxPooling2D" if ltype == "maxpool2d" else "AveragePooling2D"
            lines.append(f"    keras.layers.{cls}({k}),")
        elif ltype == "flatten":
            lines.append("    keras.layers.Flatten(),")
        elif ltype == "batchnorm":
            lines.append("    keras.layers.BatchNormalization(),")
        elif ltype in {"rnn", "lstm", "gru"}:
            hidden = layer.hidden_size or layer.neurons
            return_seq = "True" if layer.return_sequences else "False"
            cls = "SimpleRNN" if ltype == "rnn" else "LSTM" if ltype == "lstm" else "GRU"
            lines.append(f"    keras.layers.{cls}({hidden}, return_sequences={return_seq}),")
        elif ltype == "embedding":
            vocab = layer.vocab_size or 50
            emb = layer.embedding_dim or layer.neurons
            lines.append(f"    keras.layers.Embedding({vocab}, {emb}),")
        elif ltype == "attention":
            lines.append("    # Attention layer not represented in Sequential")
        elif ltype == "residual":
            lines.append("    # Residual block not represented in Sequential")

    lines.append("])")
    return "\n".join(lines)
