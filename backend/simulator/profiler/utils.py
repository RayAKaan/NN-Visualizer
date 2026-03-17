from __future__ import annotations

from typing import List, Tuple

from ..layers import LayerConfig


def _shape_size(shape: Tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def infer_layer_shapes(layers: List[LayerConfig]) -> List[Tuple[int, ...]]:
    """Infer output shape per layer (including input layer)."""
    if not layers:
        return []

    input_layer = layers[0]
    if input_layer.input_shape:
        current_shape = tuple(int(v) for v in input_layer.input_shape)
    elif input_layer.sequence_length and input_layer.neurons:
        current_shape = (int(input_layer.sequence_length), int(input_layer.neurons))
    else:
        current_shape = (int(input_layer.neurons),)

    shapes: List[Tuple[int, ...]] = [current_shape]

    for layer in layers[1:]:
        ltype = layer.layer_type
        if ltype in {"dense", "output"}:
            current_shape = (int(layer.neurons),)
        elif ltype == "conv2d":
            c_in, h_in, w_in = current_shape
            c_out = layer.filters or layer.neurons
            k = layer.kernel_size or 3
            stride = layer.stride or 1
            padding = (layer.padding or "valid").lower()
            pad = (k - 1) // 2 if padding == "same" else 0
            h_out = (h_in + 2 * pad - k) // stride + 1
            w_out = (w_in + 2 * pad - k) // stride + 1
            current_shape = (int(c_out), int(h_out), int(w_out))
        elif ltype in {"maxpool2d", "avgpool2d"}:
            c_in, h_in, w_in = current_shape
            k = layer.pool_size or layer.kernel_size or 2
            stride = layer.pool_stride or layer.stride or k
            h_out = (h_in - k) // stride + 1
            w_out = (w_in - k) // stride + 1
            current_shape = (int(c_in), int(h_out), int(w_out))
        elif ltype == "flatten":
            current_shape = (_shape_size(current_shape),)
        elif ltype == "batchnorm":
            current_shape = current_shape
        elif ltype in {"rnn", "lstm", "gru"}:
            t_len, _ = current_shape
            hidden = layer.hidden_size or layer.neurons
            if layer.return_sequences:
                current_shape = (int(t_len), int(hidden))
            else:
                current_shape = (int(hidden),)
        elif ltype == "embedding":
            t_len = current_shape[0]
            emb = layer.embedding_dim or layer.neurons
            current_shape = (int(t_len), int(emb))
        elif ltype == "attention":
            t_len, d_model = current_shape
            current_shape = (int(t_len), int(d_model))
        elif ltype == "residual":
            current_shape = current_shape
        else:
            current_shape = current_shape

        shapes.append(current_shape)

    return shapes
