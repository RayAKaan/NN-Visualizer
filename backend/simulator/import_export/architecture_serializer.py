from __future__ import annotations

from typing import Dict, List

from ..layers import LayerConfig


def serialize_architecture(name: str, layers: List[LayerConfig]) -> Dict:
    return {
        "name": name,
        "version": "1.0",
        "layers": [layer.__dict__ for layer in layers],
    }


def parse_architecture(payload: Dict) -> List[LayerConfig]:
    raw_layers = payload.get("layers", [])
    layers: List[LayerConfig] = []
    for layer in raw_layers:
        layers.append(
            LayerConfig(
                layer_type=layer.get("type") or layer.get("layer_type"),
                neurons=layer.get("neurons", 1),
                activation=layer.get("activation"),
                init=layer.get("init"),
                input_shape=layer.get("input_shape"),
                kernel_size=layer.get("kernel_size"),
                stride=layer.get("stride"),
                padding=layer.get("padding"),
                filters=layer.get("filters"),
                pool_size=layer.get("pool_size"),
                pool_stride=layer.get("pool_stride"),
                hidden_size=layer.get("hidden_size"),
                sequence_length=layer.get("sequence_length"),
                return_sequences=layer.get("return_sequences"),
                embedding_dim=layer.get("embedding_dim"),
                vocab_size=layer.get("vocab_size"),
                num_heads=layer.get("num_heads"),
            )
        )
    return layers
