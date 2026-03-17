from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..dataset_manager import dataset_manager
from ..graph_engine import build_graph
from ..layers import LayerConfig
from ..training_engine import TrainingConfig, TrainingEngine


def _parse_layers(raw_layers: List[dict]) -> List[LayerConfig]:
    layers = []
    for layer in raw_layers:
        layers.append(
            LayerConfig(
                layer_type=layer.get("type") or layer.get("layer_type"),
                neurons=layer.get("neurons"),
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


def _extract_train_test(dataset: dict):
    train = dataset.get("train")
    test = dataset.get("test")
    if isinstance(train, list):
        train_data = [(np.asarray(p["x"], dtype=np.float32), np.asarray(p["y"], dtype=np.float32)) for p in train]
        test_data = [(np.asarray(p["x"], dtype=np.float32), np.asarray(p["y"], dtype=np.float32)) for p in test]
        return train_data, test_data
    if isinstance(train, dict) and "x" in train and "y" in train:
        train_x = np.asarray(train["x"], dtype=np.float32)
        train_y = np.asarray(train["y"], dtype=np.float32)
        test_x = np.asarray(test["x"], dtype=np.float32)
        test_y = np.asarray(test["y"], dtype=np.float32)
        train_data = [(train_x[i], train_y[i]) for i in range(len(train_x))]
        test_data = [(test_x[i], test_y[i]) for i in range(len(test_x))]
        return train_data, test_data
    return [], []


@dataclass
class ComparisonRun:
    comparison_id: str
    results: Dict


class ComparisonManager:
    def __init__(self) -> None:
        self._runs: Dict[str, ComparisonRun] = {}

    def run(self, models: List[dict], dataset_id: str, epochs: int) -> Dict:
        dataset = dataset_manager.get(dataset_id)
        train_data, test_data = _extract_train_test(dataset)
        results = {"models": [], "loss_histories": {}, "dataset_id": dataset_id, "epochs": epochs}
        for model in models:
            model_id = model.get("model_id")
            layers = _parse_layers(model.get("architecture", []))
            graph = build_graph(layers)
            config_raw = model.get("config", {})
            cfg = TrainingConfig(
                epochs=epochs,
                batch_size=config_raw.get("batch_size", 16),
                learning_rate=config_raw.get("learning_rate", 0.01),
                optimizer=config_raw.get("optimizer", "adam"),
                loss_function=config_raw.get("loss_function", "bce"),
            )
            trainer = TrainingEngine(graph, cfg)
            history = []
            for _ in range(epochs):
                metrics = trainer.train_epoch(train_data, test_data, "cmp")
                history.append(metrics.__dict__)
            final = history[-1] if history else {}
            train_time_ms = sum(h.get("epoch_duration_ms", 0.0) for h in history)
            test_losses = [h.get("test_loss", 0.0) for h in history]
            convergence_epoch = int(test_losses.index(min(test_losses)) + 1) if test_losses else 0
            results["models"].append(
                {
                    "model_id": model_id,
                    "graph_id": graph.architecture_hash,
                    "final_train_loss": final.get("train_loss", 0.0),
                    "final_test_loss": final.get("test_loss", 0.0),
                    "final_train_accuracy": final.get("train_accuracy", 0.0),
                    "final_test_accuracy": final.get("test_accuracy", 0.0),
                    "convergence_epoch": convergence_epoch,
                    "total_params": graph.total_params,
                    "total_flops": graph.flops_per_sample,
                    "training_time_ms": train_time_ms,
                }
            )
            results["loss_histories"][model_id] = {
                "train": [h.get("train_loss", 0.0) for h in history],
                "test": [h.get("test_loss", 0.0) for h in history],
            }

        results["winner"] = None
        comparison_id = f"cmp-{len(self._runs)+1}"
        self._runs[comparison_id] = ComparisonRun(comparison_id, results)
        return {"comparison_id": comparison_id, **results}

    def get(self, comparison_id: str) -> Dict:
        run = self._runs.get(comparison_id)
        if not run:
            raise KeyError("Comparison not found")
        return {"comparison_id": comparison_id, **run.results}


comparison_manager = ComparisonManager()


def setup_comparison(models: List[dict], dataset_id: str, epochs: int) -> Dict:
    return comparison_manager.run(models, dataset_id, epochs)
