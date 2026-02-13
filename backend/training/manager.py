from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from model_utils import build_activation_model
from model.ann_model import build_ann
from training.gradient_engine import clip_gradients

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_store"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str
    weight_decay: float
    activation: str
    initializer: str
    dropout: float


class TrainingManager:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._step_event = threading.Event()
        self._epoch_step_event = threading.Event()
        self._config: Optional[TrainingConfig] = None
        self._emit: Optional[Callable[[dict], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._status = "idle"
        self._latest_model: Optional[tf.keras.Model] = None
        self._latest_metrics: dict = {}

    def configure(self, config: TrainingConfig) -> None:
        self._config = config

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> str:
        return self._status

    def latest_metrics(self) -> dict:
        return self._latest_metrics

    def latest_model(self) -> Optional[tf.keras.Model]:
        return self._latest_model

    def start(self, emit: Callable[[dict], None], loop: asyncio.AbstractEventLoop) -> None:
        if self.is_running():
            return
        if self._config is None:
            raise RuntimeError("Training config not set")
        self._emit = emit
        self._loop = loop
        self._stop_event.clear()
        self._pause_event.clear()
        self._step_event.clear()
        self._epoch_step_event.clear()
        self._status = "running"
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()
        self._status = "paused"

    def resume(self) -> None:
        self._pause_event.clear()
        self._status = "running"

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        self._step_event.set()
        self._epoch_step_event.set()
        self._status = "stopping"

    def step_batch(self) -> None:
        self._step_event.set()

    def step_epoch(self) -> None:
        self._epoch_step_event.set()
        self._step_event.set()

    def save_latest(self, name: str) -> str:
        if self._latest_model is None:
            raise RuntimeError("No trained model available to save")
        out_path = MODELS_DIR / f"{name}.keras"
        self._latest_model.save(out_path)
        return str(out_path)

    def list_models(self) -> list[dict]:
        out = []
        for path in MODELS_DIR.glob("*.keras"):
            out.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "created_at": path.stat().st_mtime,
                    "dataset": "MNIST",
                    "architecture": "ANN 784-128-64-10",
                }
            )
        return sorted(out, key=lambda x: x["created_at"], reverse=True)

    def delete_model(self, name: str) -> None:
        path = MODELS_DIR / f"{name}.keras"
        if path.exists():
            path.unlink()


    def load_saved_model(self, name: str) -> tf.keras.Model:
        path = MODELS_DIR / f"{name}.keras"
        if not path.exists():
            raise FileNotFoundError(f"Saved model not found: {name}")
        return tf.keras.models.load_model(path)
    def _emit_message(self, message: dict) -> None:
        if not self._emit or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._emit(message), self._loop)

    def _build_model(self) -> tf.keras.Model:
        assert self._config is not None
        return build_ann(
            activation=self._config.activation,
            initializer=self._config.initializer,
            weight_decay=self._config.weight_decay,
            dropout=self._config.dropout,
        )

    def _optimizer(self) -> tf.keras.optimizers.Optimizer:
        assert self._config is not None
        lr = self._config.learning_rate
        if self._config.optimizer == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=lr)
        if self._config.optimizer == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=lr)
        return tf.keras.optimizers.Adam(learning_rate=lr)

    def _run_training(self) -> None:
        assert self._config is not None
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

        split_idx = int(len(x_train) * 0.9)
        x_val, y_val = x_train[split_idx:], y_train[split_idx:]
        x_train, y_train = x_train[:split_idx], y_train[:split_idx]

        model = self._build_model()
        optimizer = self._optimizer()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        activation_model = build_activation_model(model)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(
            self._config.batch_size
        )
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self._config.batch_size)

        for epoch in range(1, self._config.epochs + 1):
            if self._stop_event.is_set():
                break

            epoch_losses: list[float] = []
            epoch_accs: list[float] = []

            for batch_index, (batch_x, batch_y) in enumerate(dataset):
                if self._stop_event.is_set():
                    break

                if self._pause_event.is_set():
                    self._step_event.wait()
                    self._step_event.clear()

                with tf.GradientTape() as tape:
                    logits = model(batch_x, training=True)
                    loss_value = loss_fn(batch_y, logits) + sum(model.losses)

                grads = tape.gradient(loss_value, model.trainable_variables)
                grads = clip_gradients(grads, clip_value=5.0)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                preds = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(preds == batch_y, tf.float32))

                epoch_losses.append(float(loss_value.numpy()))
                epoch_accs.append(float(accuracy.numpy()))

                activations = activation_model.predict(batch_x[:1], verbose=0)
                weights = model.get_weights()
                grad_h1_h2 = grads[2].numpy() if grads[2] is not None else np.zeros_like(weights[2])
                grad_h2_out = grads[4].numpy() if grads[4] is not None else np.zeros_like(weights[4])

                self._emit_message(
                    {
                        "type": "batch_update",
                        "epoch": epoch,
                        "batch": batch_index,
                        "activations": {
                            "hidden1": activations[0][0].tolist(),
                            "hidden2": activations[1][0].tolist(),
                            "output": activations[2][0].tolist(),
                        },
                        "gradients": {
                            "hidden1_hidden2": grad_h1_h2.tolist(),
                            "hidden2_output": grad_h2_out.tolist(),
                        },
                        "weights": {
                            "hidden1_hidden2": weights[2].tolist(),
                            "hidden2_output": weights[4].tolist(),
                        },
                        "loss": float(loss_value.numpy()),
                        "accuracy": float(accuracy.numpy()),
                        "learning_rate": float(optimizer.learning_rate.numpy()),
                        "gradient_norm": float(
                            np.sqrt(np.mean(np.square(grad_h1_h2)) + np.mean(np.square(grad_h2_out)))
                        ),
                        "timestamp": time.time(),
                    }
                )

                if batch_index % 20 == 0:
                    self._emit_message(
                        {
                            "type": "weights_update",
                            "epoch": epoch,
                            "weights": {
                                "hidden1_hidden2": weights[2].tolist(),
                                "hidden2_output": weights[4].tolist(),
                            },
                        }
                    )

                if self._epoch_step_event.is_set():
                    self._epoch_step_event.clear()
                    break

            val_losses: list[float] = []
            val_accs: list[float] = []
            for batch_x, batch_y in val_dataset:
                val_logits = model(batch_x, training=False)
                val_loss = loss_fn(batch_y, val_logits)
                val_preds = tf.argmax(val_logits, axis=1)
                val_acc = tf.reduce_mean(tf.cast(val_preds == batch_y, tf.float32))
                val_losses.append(float(val_loss.numpy()))
                val_accs.append(float(val_acc.numpy()))

            epoch_msg = {
                "type": "epoch_update",
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "accuracy": float(np.mean(epoch_accs)) if epoch_accs else 0.0,
                "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
                "val_accuracy": float(np.mean(val_accs)) if val_accs else 0.0,
            }
            self._latest_metrics = epoch_msg
            self._emit_message(epoch_msg)

            if self._pause_event.is_set():
                self._status = "paused"

        self._latest_model = model
        if self._stop_event.is_set():
            self._status = "stopped"
            self._emit_message({"type": "training_stopped"})
        else:
            self._status = "completed"
            self._emit_message({"type": "training_complete", "metrics": self._latest_metrics})
