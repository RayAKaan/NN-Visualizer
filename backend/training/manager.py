from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from model_utils import build_activation_model


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
        self._lock = threading.Lock()
        self._config: Optional[TrainingConfig] = None
        self._emit: Optional[Callable[[dict], None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def configure(self, config: TrainingConfig) -> None:
        with self._lock:
            self._config = config

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause_event.set()

    def resume(self) -> None:
        self._pause_event.clear()

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        self._step_event.set()
        self._epoch_step_event.set()

    def step_batch(self) -> None:
        self._step_event.set()

    def step_epoch(self) -> None:
        self._epoch_step_event.set()

    def _emit_message(self, message: dict) -> None:
        if not self._emit or not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._emit(message), self._loop)

    def _build_model(self) -> tf.keras.Model:
        assert self._config is not None
        activation_name = self._config.activation
        initializer = tf.keras.initializers.get(self._config.initializer)
        dropout = self._config.dropout
        activation_layer = (
            tf.keras.layers.LeakyReLU() if activation_name == "leaky_relu" else activation_name
        )
        layers = [
            tf.keras.layers.Input(shape=(784,)),
            tf.keras.layers.Dense(
                128,
                activation=activation_layer,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.L2(self._config.weight_decay),
            ),
        ]
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(
            tf.keras.layers.Dense(
                64,
                activation=activation_layer,
                kernel_initializer=initializer,
                kernel_regularizer=tf.keras.regularizers.L2(self._config.weight_decay),
            )
        )
        if dropout > 0:
            layers.append(tf.keras.layers.Dropout(dropout))
        layers.append(
            tf.keras.layers.Dense(
                10,
                activation="softmax",
                kernel_initializer=initializer,
            )
        )
        model = tf.keras.Sequential(layers)
        return model

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

        model = self._build_model()
        optimizer = self._optimizer()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        activation_model = build_activation_model(model)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(10000).batch(self._config.batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
            self._config.batch_size
        )

        for epoch in range(1, self._config.epochs + 1):
            if self._stop_event.is_set():
                break

            epoch_losses = []
            epoch_accuracy = []
            batch_index = 0

            for batch_x, batch_y in dataset:
                if self._stop_event.is_set():
                    break
                if self._pause_event.is_set():
                    self._step_event.wait()
                    self._step_event.clear()

                with tf.GradientTape() as tape:
                    logits = model(batch_x, training=True)
                    loss_value = loss_fn(batch_y, logits)
                    loss_value += sum(model.losses)

                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(predictions == batch_y, tf.float32))

                epoch_losses.append(loss_value.numpy())
                epoch_accuracy.append(accuracy.numpy())

                activations = activation_model.predict(batch_x[:1])
                weights = model.get_weights()
                gradients = {
                    "hidden1_hidden2": grads[2].numpy().tolist(),
                    "hidden2_output": grads[4].numpy().tolist(),
                }

                self._emit_message(
                    {
                        "type": "batch",
                        "epoch": epoch,
                        "batch": batch_index,
                        "loss": float(loss_value.numpy()),
                        "accuracy": float(accuracy.numpy()),
                        "activations": {
                            "hidden1": activations[0][0].tolist(),
                            "hidden2": activations[1][0].tolist(),
                            "output": activations[2][0].tolist(),
                        },
                        "gradients": gradients,
                    }
                )

                if batch_index % 20 == 0:
                    self._emit_message(
                        {
                            "type": "weights",
                            "epoch": epoch,
                            "weights": {
                                "hidden1_hidden2": weights[2].tolist(),
                                "hidden2_output": weights[4].tolist(),
                            },
                        }
                    )

                batch_index += 1

                if self._epoch_step_event.is_set():
                    self._epoch_step_event.clear()
                    break

            val_losses = []
            val_accuracies = []
            for batch_x, batch_y in val_dataset:
                logits = model(batch_x, training=False)
                loss_value = loss_fn(batch_y, logits)
                predictions = tf.argmax(logits, axis=1)
                accuracy = tf.reduce_mean(tf.cast(predictions == batch_y, tf.float32))
                val_losses.append(loss_value.numpy())
                val_accuracies.append(accuracy.numpy())

            self._emit_message(
                {
                    "type": "epoch",
                    "epoch": epoch,
                    "loss": float(np.mean(epoch_losses)),
                    "accuracy": float(np.mean(epoch_accuracy)),
                    "val_loss": float(np.mean(val_losses)),
                    "val_accuracy": float(np.mean(val_accuracies)),
                }
            )

            if self._epoch_step_event.is_set():
                self._epoch_step_event.clear()
                self._pause_event.set()

        self._emit_message({"type": "stopped"})
