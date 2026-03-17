from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LayerConfig:
    layer_type: str
    neurons: int
    activation: str | None = None
    init: str | None = None
    input_shape: Optional[List[int]] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = None
    padding: Optional[str] = None
    filters: Optional[int] = None
    pool_size: Optional[int] = None
    pool_stride: Optional[int] = None
    hidden_size: Optional[int] = None
    sequence_length: Optional[int] = None
    return_sequences: Optional[bool] = None
    embedding_dim: Optional[int] = None
    vocab_size: Optional[int] = None
    num_heads: Optional[int] = None


@dataclass
class ValidationResult:
    valid: bool
    architecture: List[int]
    activations: List[str]
    total_params: int
    flops_per_sample: int
    layer_params: List[dict]
    errors: List[str]
    warnings: List[str]


def _is_supported(layer_type: str) -> bool:
    return layer_type in {
        "dense",
        "input",
        "output",
        "conv2d",
        "maxpool2d",
        "avgpool2d",
        "flatten",
        "batchnorm",
        "rnn",
        "lstm",
        "gru",
        "embedding",
        "attention",
        "residual",
    }


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def validate_layers(layers: List[LayerConfig]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    architecture: List[int] = []
    activations: List[str] = []
    layer_params: List[dict] = []

    if len(layers) < 2:
        errors.append("At least input and output layers are required.")
    if len(layers) > 12:
        errors.append("Maximum 10 hidden layers allowed.")

    if layers:
        if layers[0].layer_type != "input":
            errors.append("First layer must be input.")
        if layers[-1].layer_type != "output":
            errors.append("Last layer must be output.")

    total_params = 0
    flops = 0

    if not layers:
        return ValidationResult(
            valid=False,
            architecture=[],
            activations=[],
            total_params=0,
            flops_per_sample=0,
            layer_params=[],
            errors=errors,
            warnings=warnings,
        )

    input_layer = layers[0]
    if input_layer.input_shape:
        current_shape = tuple(int(v) for v in input_layer.input_shape)
    else:
        current_shape = (input_layer.neurons,)
    if len(current_shape) == 3:
        current_type = "spatial"
    elif len(current_shape) == 2:
        current_type = "sequence"
    else:
        current_type = "vector"

    for idx, layer in enumerate(layers):
        if layer.neurons < 1 or layer.neurons > 512:
            errors.append(f"Layer {idx} has invalid neuron count (1-512).")
        if not _is_supported(layer.layer_type):
            errors.append(f"Layer {idx} type '{layer.layer_type}' not supported.")

        if idx == 0:
            architecture.append(layer.neurons)
            continue

        ltype = layer.layer_type
        if ltype in {"dense", "output"}:
            if current_type != "vector":
                errors.append(f"Layer {idx} ({ltype}) requires vector input. Add Flatten first.")
            in_dim = current_shape[0] if current_shape else layer.neurons
            out_dim = layer.neurons
            weights = in_dim * out_dim
            biases = out_dim
            total = weights + biases
            total_params += total
            layer_params.append({"layer": idx, "weights": weights, "biases": biases, "total": total})
            flops += 2 * weights + biases
            current_shape = (out_dim,)
            current_type = "vector"
            activations.append((layer.activation or "linear").lower())
            architecture.append(out_dim)
        elif ltype == "conv2d":
            if current_type != "spatial":
                errors.append(f"Layer {idx} (conv2d) requires spatial input (C,H,W).")
                continue
            c_in, h_in, w_in = current_shape
            c_out = layer.filters or layer.neurons
            k = layer.kernel_size or 3
            stride = layer.stride or 1
            padding = (layer.padding or "valid").lower()
            pad = (k - 1) // 2 if padding == "same" else 0
            h_out = (h_in + 2 * pad - k) // stride + 1
            w_out = (w_in + 2 * pad - k) // stride + 1
            weights = c_out * c_in * k * k
            biases = c_out
            total = weights + biases
            total_params += total
            layer_params.append({"layer": idx, "weights": weights, "biases": biases, "total": total})
            flops += 2 * c_in * k * k * c_out * h_out * w_out
            current_shape = (c_out, h_out, w_out)
            current_type = "spatial"
            activations.append((layer.activation or "linear").lower())
            architecture.append(_shape_size(current_shape))
        elif ltype in {"maxpool2d", "avgpool2d"}:
            if current_type != "spatial":
                errors.append(f"Layer {idx} ({ltype}) requires spatial input (C,H,W).")
                continue
            c_in, h_in, w_in = current_shape
            k = layer.pool_size or layer.kernel_size or 2
            stride = layer.pool_stride or layer.stride or k
            h_out = (h_in - k) // stride + 1
            w_out = (w_in - k) // stride + 1
            layer_params.append({"layer": idx, "weights": 0, "biases": 0, "total": 0})
            flops += c_in * h_out * w_out * k * k
            current_shape = (c_in, h_out, w_out)
            current_type = "spatial"
            architecture.append(_shape_size(current_shape))
        elif ltype == "flatten":
            if current_type != "spatial":
                errors.append(f"Layer {idx} (flatten) requires spatial input.")
                continue
            current_shape = (_shape_size(current_shape),)
            current_type = "vector"
            layer_params.append({"layer": idx, "weights": 0, "biases": 0, "total": 0})
            architecture.append(current_shape[0])
        elif ltype == "batchnorm":
            if current_type == "vector":
                features = current_shape[0]
                weights = features
                biases = features
            elif current_type == "spatial":
                features = current_shape[0]
                weights = features
                biases = features
            else:
                errors.append(f"Layer {idx} (batchnorm) requires vector or spatial input.")
                continue
            total = weights + biases
            total_params += total
            layer_params.append({"layer": idx, "weights": weights, "biases": biases, "total": total})
            architecture.append(_shape_size(current_shape))
        elif ltype in {"rnn", "lstm", "gru"}:
            if current_type != "sequence":
                errors.append(f"Layer {idx} ({ltype}) requires sequence input (T,D).")
                continue
            t_len, d_in = current_shape
            hidden = layer.hidden_size or layer.neurons
            if ltype == "rnn":
                total = hidden * (d_in + hidden + 1)
                total_params += total
                layer_params.append({"layer": idx, "weights": hidden * (d_in + hidden), "biases": hidden, "total": total})
            else:
                if ltype == "gru":
                    total = 3 * hidden * (d_in + hidden + 1)
                    total_params += total
                    layer_params.append({"layer": idx, "weights": 3 * hidden * (d_in + hidden), "biases": 3 * hidden, "total": total})
                else:
                    total = 4 * hidden * (d_in + hidden + 1)
                    total_params += total
                    layer_params.append({"layer": idx, "weights": 4 * hidden * (d_in + hidden), "biases": 4 * hidden, "total": total})
            return_seq = bool(layer.return_sequences)
            if return_seq:
                current_shape = (t_len, hidden)
                current_type = "sequence"
                architecture.append(t_len * hidden)
            else:
                current_shape = (hidden,)
                current_type = "vector"
                architecture.append(hidden)
        elif ltype == "embedding":
            if current_type not in {"sequence", "vector"}:
                errors.append(f"Layer {idx} (embedding) requires sequence input.")
                continue
            t_len = current_shape[0]
            vocab = layer.vocab_size or 50
            emb = layer.embedding_dim or layer.neurons
            total = vocab * emb
            total_params += total
            layer_params.append({"layer": idx, "weights": total, "biases": 0, "total": total})
            current_shape = (t_len, emb)
            current_type = "sequence"
            architecture.append(t_len * emb)
        elif ltype == "attention":
            if current_type != "sequence":
                errors.append(f"Layer {idx} (attention) requires sequence input.")
                continue
            t_len, d_model = current_shape
            total = 4 * d_model * d_model
            total_params += total
            layer_params.append({"layer": idx, "weights": total, "biases": 0, "total": total})
            current_shape = (t_len, d_model)
            current_type = "sequence"
            architecture.append(t_len * d_model)
        elif ltype == "residual":
            layer_params.append({"layer": idx, "weights": 0, "biases": 0, "total": 0})
            architecture.append(_shape_size(current_shape))

    return ValidationResult(
        valid=len(errors) == 0,
        architecture=architecture,
        activations=activations,
        total_params=total_params,
        flops_per_sample=flops,
        layer_params=layer_params,
        errors=errors,
        warnings=warnings,
    )


__all__ = ["LayerConfig", "ValidationResult", "validate_layers"]
