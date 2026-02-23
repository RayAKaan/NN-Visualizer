import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import tensorflow as tf

from model import MODEL_BUILDERS
from training.gradient_engine import clip_gradients
import config


@dataclass
class TrainingConfig:
    model_type: str = "ann"
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    optimizer: str = "adam"
    activation: str = "relu"
    weight_decay: float = 0.0
    dropout_rate: float = 0.2
    kernel_initializer: str = "glorot_uniform"
    lstm_units: int = 128
    bidirectional: bool = False
    conv1_filters: int = 32
    conv2_filters: int = 64


class TrainingManager:
    def __init__(self):
        self.config = TrainingConfig()
        self.status = "idle"
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self._ws_callback: Callable[[dict], None] | None = None
        self._thread = None
        self._pause_flag = threading.Event()
        self._stop_flag = threading.Event()
        self.batch_history = []
        self.epoch_history = []

    def configure(self, cfg: dict):
        self.config = TrainingConfig(**{**asdict(self.config), **cfg})
        return self.get_status()

    def start(self, callback: Callable[[dict], None] | None = None):
        if self.status == "running":
            return
        self._ws_callback = callback
        self._stop_flag.clear()
        self._pause_flag.clear()
        self.status = "running"
        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()

    def pause(self):
        if self.status == "running":
            self.status = "paused"
            self._pause_flag.set()

    def resume(self):
        if self.status == "paused":
            self.status = "running"
            self._pause_flag.clear()

    def stop(self):
        self.status = "stopping"
        self._stop_flag.set()

    def step_batch(self):
        return {"type": "info", "message": "step_batch available during websocket-driven mode"}

    def step_epoch(self):
        return {"type": "info", "message": "step_epoch available during websocket-driven mode"}

    def _emit(self, payload: dict):
        if self._ws_callback:
            self._ws_callback(payload)

    def _reshape(self, x):
        if self.config.model_type == "ann":
            return x.reshape((-1, 784))
        if self.config.model_type == "cnn":
            return x.reshape((-1, 28, 28, 1))
        return x.reshape((-1, 28, 28))

    def _build_optimizer(self):
        opt = self.config.optimizer.lower()
        if opt == "sgd":
            return tf.keras.optimizers.SGD(self.config.learning_rate)
        if opt == "rmsprop":
            return tf.keras.optimizers.RMSprop(self.config.learning_rate)
        return tf.keras.optimizers.Adam(self.config.learning_rate)

    def _compute_class_metrics(self, y_true, y_pred):
        cm = np.zeros((10, 10), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        precision, recall, f1 = [], [], []
        for i in range(10):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            pr = tp / (tp + fp) if tp + fp > 0 else 0.0
            rc = tp / (tp + fn) if tp + fn > 0 else 0.0
            f = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
            precision.append(float(pr)); recall.append(float(rc)); f1.append(float(f))
        return cm.tolist(), precision, recall, f1

    def _training_loop(self):
        try:
            (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
            x_train = self._reshape(x_train.astype("float32") / 255.0)
            x_val = self._reshape(x_val.astype("float32") / 255.0)
            kwargs = {}
            if self.config.model_type == "ann":
                kwargs.update(dropout_rate=self.config.dropout_rate, activation=self.config.activation,
                              kernel_initializer=self.config.kernel_initializer, l2_reg=self.config.weight_decay)
            elif self.config.model_type == "cnn":
                kwargs.update(dropout_rate=self.config.dropout_rate, activation=self.config.activation,
                              conv1_filters=self.config.conv1_filters, conv2_filters=self.config.conv2_filters)
            else:
                kwargs.update(dropout_rate=self.config.dropout_rate, lstm_units=self.config.lstm_units,
                              bidirectional=self.config.bidirectional)
            model = MODEL_BUILDERS[self.config.model_type](**kwargs)
            opt = self._build_optimizer()
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

            ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.config.batch_size)
            self.total_batches = int(np.ceil(len(x_train) / self.config.batch_size))

            for epoch in range(1, self.config.epochs + 1):
                if self._stop_flag.is_set():
                    self.status = "stopped"
                    self._emit({"type": "training_stopped"})
                    return
                self.current_epoch = epoch
                for batch_idx, (xb, yb) in enumerate(ds, start=1):
                    while self._pause_flag.is_set() and not self._stop_flag.is_set():
                        time.sleep(0.1)
                    if self._stop_flag.is_set():
                        self.status = "stopped"
                        self._emit({"type": "training_stopped"})
                        return
                    with tf.GradientTape() as tape:
                        preds = model(xb, training=True)
                        loss = loss_fn(yb, preds)
                    grads = tape.gradient(loss, model.trainable_variables)
                    clipped_grads, norm = clip_gradients(grads)
                    opt.apply_gradients(zip(clipped_grads, model.trainable_variables))
                    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(preds, axis=1), tf.cast(yb, tf.int64)), tf.float32))
                    self.current_batch = batch_idx

                    batch_msg = {
                        "type": "batch_update", "epoch": epoch, "batch": batch_idx,
                        "total_batches": self.total_batches, "total_epochs": self.config.epochs,
                        "model_type": self.config.model_type, "loss": float(loss.numpy()), "accuracy": float(acc.numpy()),
                        "learning_rate": float(tf.keras.backend.get_value(opt.learning_rate)),
                        "gradient_norm": norm,
                        "activations": {l.name: {"type": l.__class__.__name__, "shape": str(getattr(l, 'output_shape', ''))} for l in model.layers},
                        "gradients": {v.name: {"norm": float(tf.norm(g).numpy()) if g is not None else 0.0,
                                                "mean": float(tf.reduce_mean(g).numpy()) if g is not None else 0.0,
                                                "std": float(tf.math.reduce_std(g).numpy()) if g is not None else 0.0,
                                                "max_abs": float(tf.reduce_max(tf.abs(g)).numpy()) if g is not None else 0.0,
                                                "shape": str(v.shape)} for g, v in zip(clipped_grads, model.trainable_variables)},
                        "weights": None,
                        "timestamp": time.time(),
                    }
                    self.batch_history.append(batch_msg)
                    self._emit(batch_msg)

                val_preds = model.predict(x_val, verbose=0)
                val_classes = np.argmax(val_preds, axis=1)
                val_loss = loss_fn(y_val, val_preds).numpy()
                val_acc = float((val_classes == y_val).mean())
                cm, pr, rc, f1 = self._compute_class_metrics(y_val, val_classes)
                epoch_msg = {
                    "type": "epoch_update", "epoch": epoch, "total_epochs": self.config.epochs,
                    "model_type": self.config.model_type, "loss": float(self.batch_history[-1]["loss"]),
                    "accuracy": float(self.batch_history[-1]["accuracy"]), "val_loss": float(val_loss),
                    "val_accuracy": val_acc, "precision_per_class": pr, "recall_per_class": rc,
                    "f1_per_class": f1, "confusion_matrix": cm, "timestamp": time.time(),
                }
                self.epoch_history.append(epoch_msg)
                self._emit(epoch_msg)

            self.status = "completed"
            self._emit({"type": "training_complete", "model_type": self.config.model_type})
        except Exception as exc:
            self.status = "error"
            self._emit({"type": "training_error", "error": str(exc)})

    def get_status(self):
        return {
            "status": self.status,
            "epoch": self.current_epoch,
            "batch": self.current_batch,
            "total_batches": self.total_batches,
            "config": asdict(self.config),
        }

    def get_training_history(self):
        return self.batch_history

    def get_epoch_history(self):
        return self.epoch_history


training_manager = TrainingManager()
