from __future__ import annotations

from typing import Dict, List


def list_templates() -> List[Dict]:
    return [
        {
            "id": "simple_mlp",
            "name": "Simple MLP",
            "category": "2D Classification",
            "layers": [
                {"type": "input", "neurons": 2},
                {"type": "dense", "neurons": 8, "activation": "relu", "init": "xavier"},
                {"type": "output", "neurons": 1, "activation": "sigmoid", "init": "xavier"},
            ],
        },
        {
            "id": "simple_cnn",
            "name": "Simple CNN",
            "category": "Image Classification",
            "layers": [
                {"type": "input", "neurons": 1, "input_shape": [1, 28, 28]},
                {"type": "conv2d", "neurons": 8, "filters": 8, "kernel_size": 3, "stride": 1, "padding": "same", "activation": "relu"},
                {"type": "maxpool2d", "neurons": 1, "pool_size": 2, "pool_stride": 2},
                {"type": "flatten", "neurons": 1},
                {"type": "dense", "neurons": 32, "activation": "relu"},
                {"type": "output", "neurons": 10, "activation": "softmax"},
            ],
        },
        {
            "id": "lenet5",
            "name": "LeNet-5",
            "category": "Image Classification",
            "layers": [
                {"type": "input", "neurons": 1, "input_shape": [1, 28, 28]},
                {"type": "conv2d", "neurons": 6, "filters": 6, "kernel_size": 5, "stride": 1, "padding": "valid", "activation": "relu"},
                {"type": "avgpool2d", "neurons": 1, "pool_size": 2, "pool_stride": 2},
                {"type": "conv2d", "neurons": 16, "filters": 16, "kernel_size": 5, "stride": 1, "padding": "valid", "activation": "relu"},
                {"type": "avgpool2d", "neurons": 1, "pool_size": 2, "pool_stride": 2},
                {"type": "flatten", "neurons": 1},
                {"type": "dense", "neurons": 120, "activation": "relu"},
                {"type": "dense", "neurons": 84, "activation": "relu"},
                {"type": "output", "neurons": 10, "activation": "softmax"},
            ],
        },
        {
            "id": "lstm_classifier",
            "name": "LSTM Classifier",
            "category": "Sequence",
            "layers": [
                {"type": "input", "neurons": 1, "input_shape": [20, 1]},
                {"type": "lstm", "neurons": 32, "hidden_size": 32, "return_sequences": False},
                {"type": "output", "neurons": 3, "activation": "softmax"},
            ],
        },
        {
            "id": "attention_classifier",
            "name": "Attention Classifier",
            "category": "Sequence",
            "layers": [
                {"type": "input", "neurons": 1, "input_shape": [10, 8]},
                {"type": "attention", "neurons": 8, "num_heads": 2},
                {"type": "flatten", "neurons": 1},
                {"type": "output", "neurons": 5, "activation": "softmax"},
            ],
        },
    ]
