from __future__ import annotations

from typing import Dict, List

from ..graph_engine import NetworkGraph
from .utils import infer_layer_shapes


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def compute_flops(graph: NetworkGraph) -> Dict:
    layers = graph.layers
    shapes = infer_layer_shapes(layers)
    per_layer: List[Dict] = []
    total_forward = 0

    for idx, layer in enumerate(layers[1:], start=1):
        ltype = layer.layer_type
        input_shape = shapes[idx - 1]
        output_shape = shapes[idx]
        params = 0
        flops_fwd = 0

        if ltype in {"dense", "output"}:
            in_dim = int(input_shape[0])
            out_dim = int(output_shape[0])
            params = in_dim * out_dim + out_dim
            flops_fwd = 2 * in_dim * out_dim + out_dim
        elif ltype == "conv2d":
            c_in, h_in, w_in = input_shape
            c_out, h_out, w_out = output_shape
            k = layer.kernel_size or 3
            params = c_out * c_in * k * k + c_out
            flops_fwd = 2 * c_in * k * k * c_out * h_out * w_out
        elif ltype in {"maxpool2d", "avgpool2d"}:
            c_out, h_out, w_out = output_shape
            k = layer.pool_size or layer.kernel_size or 2
            flops_fwd = c_out * h_out * w_out * k * k
        elif ltype == "flatten":
            flops_fwd = 0
        elif ltype == "batchnorm":
            features = output_shape[0]
            params = 2 * features
            flops_fwd = features * 4
        elif ltype in {"rnn", "lstm", "gru"}:
            if len(input_shape) == 1:
                t_len = 1
                d_in = input_shape[0]
            else:
                t_len, d_in = input_shape
            hidden = layer.hidden_size or layer.neurons
            if ltype == "rnn":
                params = hidden * (d_in + hidden + 1)
                flops_fwd = t_len * hidden * (d_in + hidden)
            elif ltype == "gru":
                params = 3 * hidden * (d_in + hidden + 1)
                flops_fwd = 3 * t_len * hidden * (d_in + hidden)
            else:
                params = 4 * hidden * (d_in + hidden + 1)
                flops_fwd = 4 * t_len * hidden * (d_in + hidden)
        elif ltype == "embedding":
            vocab = layer.vocab_size or 50
            emb = layer.embedding_dim or layer.neurons
            params = vocab * emb
            flops_fwd = 0
        elif ltype == "attention":
            t_len, d_model = input_shape
            params = 4 * d_model * d_model
            flops_fwd = 2 * t_len * d_model * d_model + 2 * t_len * t_len * d_model
        elif ltype == "residual":
            flops_fwd = _shape_size(output_shape)

        total_forward += flops_fwd
        if ltype == "conv2d":
            flops_bwd = int(flops_fwd * 3)
        else:
            flops_bwd = int(flops_fwd * 2)

        per_layer.append(
            {
                "layer_index": idx,
                "layer_type": ltype,
                "params": int(params),
                "flops_forward": int(flops_fwd),
                "flops_backward": int(flops_bwd),
                "output_shape": [int(v) for v in output_shape],
            }
        )

    return {
        "total_flops_forward": int(total_forward),
        "total_flops_backward": int(total_forward * 2),
        "per_layer": per_layer,
    }
