import threading
import time
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class TrainingConfig:
    model_type: str = "ann"
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adam"
    activation: str = "relu"
    weight_decay: float = 0.0
    dropout_rate: float = 0.0
    kernel_initializer: str = "glorot_uniform"


class TrainingManager:
    def __init__(self):
        self.status: str = "idle"
        self.config = TrainingConfig()
        self.training_history: list[dict] = []
        self.epoch_history: list[dict] = []
        self.current_epoch: int = 0
        self.current_batch: int = 0
        self.total_batches: int = 0
        self.total_epochs: int = 0
        self.ws_callback: Optional[Callable[[dict], Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._stop_flag = False
        self._step_batch_flag = False
        self._step_epoch_flag = False
        self.model: Optional[tf.keras.Model] = None
        self.optimizer_instance = None

    def configure(self, config_dict: dict) -> dict:
        """Update training configuration. Only allowed when idle or completed."""
        if self.status in ("running", "paused"):
            return {"error": "Cannot configure while training is active"}

        for key, val in config_dict.items():
            if hasattr(self.config, key):
                current = getattr(self.config, key)
                if isinstance(current, bool):
                    casted = bool(val)
                elif isinstance(current, int):
                    casted = int(val)
                elif isinstance(current, float):
                    casted = float(val)
                else:
                    casted = str(val)
                setattr(self.config, key, casted)

        return {"status": "configured", "config": self.config.__dict__}

    def start(self) -> dict:
        """Start training in a background thread."""
        if self.status in ("running", "paused"):
            return {"error": "Training already active"}

        self._stop_flag = False
        self._step_batch_flag = False
        self._step_epoch_flag = False
        self._pause_event.set()
        self.training_history = []
        self.epoch_history = []
        self.current_epoch = 0
        self.current_batch = 0
        self.status = "running"
        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()
        return {"status": "started"}

    def pause(self) -> dict:
        if self.status == "running":
            self._pause_event.clear()
            self.status = "paused"
        return {"status": self.status}

    def resume(self) -> dict:
        if self.status == "paused":
            self._pause_event.set()
            self.status = "running"
        return {"status": self.status}

    def stop(self) -> dict:
        if self.status in ("running", "paused"):
            self._stop_flag = True
            self._pause_event.set()
            self.status = "stopping"
        return {"status": self.status}

    def step_batch(self) -> dict:
        """Execute exactly one batch then pause."""
        self._step_batch_flag = True
        if self.status == "paused":
            self._pause_event.set()
            self.status = "running"
        return {"status": "step_batch"}

    def step_epoch(self) -> dict:
        """Execute until end of current epoch then pause."""
        self._step_epoch_flag = True
        if self.status == "paused":
            self._pause_event.set()
            self.status = "running"
        return {"status": "step_epoch"}

    def get_status(self) -> dict:
        return {
            "status": self.status,
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "batch": self.current_batch,
            "total_batches": self.total_batches,
            "config": self.config.__dict__,
            "history_length": len(self.training_history),
            "epoch_history_length": len(self.epoch_history),
        }

    def get_training_history(self) -> list[dict]:
        return self.training_history

    def get_epoch_history(self) -> list[dict]:
        return self.epoch_history

    def _training_loop(self):
        """Main training loop — runs in background thread."""
        try:
            from training.gradient_engine import clip_gradients

            (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_val = x_val.astype("float32") / 255.0
            y_train = y_train.astype("int64")
            y_val = y_val.astype("int64")

            if self.config.model_type == "ann":
                from model.ann_model import build_ann

                x_train = x_train.reshape(-1, 784)
                x_val = x_val.reshape(-1, 784)
                self.model = build_ann(
                    activation=self.config.activation,
                    initializer=self.config.kernel_initializer,
                    weight_decay=self.config.weight_decay,
                    dropout=self.config.dropout_rate,
                )
            elif self.config.model_type == "cnn":
                from model.cnn_model import build_cnn_model

                x_train = x_train.reshape(-1, 28, 28, 1)
                x_val = x_val.reshape(-1, 28, 28, 1)
                self.model = build_cnn_model(
                    activation=self.config.activation,
                    dropout_rate=self.config.dropout_rate,
                    kernel_initializer=self.config.kernel_initializer,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")

            lr = self.config.learning_rate
            opt_name = self.config.optimizer.lower()
            if opt_name == "adam":
                self.optimizer_instance = tf.keras.optimizers.Adam(learning_rate=lr)
            elif opt_name == "sgd":
                self.optimizer_instance = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
            elif opt_name == "rmsprop":
                self.optimizer_instance = tf.keras.optimizers.RMSprop(learning_rate=lr)
            else:
                self.optimizer_instance = tf.keras.optimizers.Adam(learning_rate=lr)

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
            batch_size = self.config.batch_size
            n_samples = x_train.shape[0]
            self.total_batches = n_samples // batch_size
            self.total_epochs = self.config.epochs

            for epoch in range(self.config.epochs):
                if self._stop_flag:
                    break

                self.current_epoch = epoch + 1

                indices = np.random.permutation(n_samples)
                x_shuffled = x_train[indices]
                y_shuffled = y_train[indices]

                epoch_losses = []
                epoch_accs = []

                for batch_idx in range(self.total_batches):
                    if self._stop_flag:
                        break

                    self._pause_event.wait()
                    if self._stop_flag:
                        break

                    self.current_batch = batch_idx + 1

                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    x_batch = x_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    x_tensor = tf.constant(x_batch)
                    y_tensor = tf.constant(y_batch)

                    with tf.GradientTape() as tape:
                        predictions = self.model(x_tensor, training=True)
                        loss = loss_fn(y_tensor, predictions)

                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    gradients = clip_gradients(gradients, clip_value=1.0)

                    self.optimizer_instance.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )

                    batch_loss = float(loss.numpy())
                    pred_classes = tf.argmax(predictions, axis=1).numpy()
                    batch_acc = float(np.mean(pred_classes == y_batch))

                    epoch_losses.append(batch_loss)
                    epoch_accs.append(batch_acc)

                    sample = x_batch[0:1]
                    sample_activations = self._extract_activations(sample)
                    gradient_info = self._extract_gradients(gradients)

                    weight_snapshot = None
                    if batch_idx % 100 == 0 or batch_idx == self.total_batches - 1:
                        weight_snapshot = self._extract_weights()

                    snapshot = {
                        "type": "batch_update",
                        "epoch": self.current_epoch,
                        "batch": self.current_batch,
                        "total_batches": self.total_batches,
                        "total_epochs": self.total_epochs,
                        "loss": round(batch_loss, 6),
                        "accuracy": round(batch_acc, 4),
                        "learning_rate": float(lr),
                        "gradient_norm": gradient_info.get("total_norm", 0),
                        "activations": sample_activations,
                        "gradients": gradient_info,
                        "timestamp": time.time(),
                    }

                    if weight_snapshot is not None:
                        snapshot["weights"] = weight_snapshot

                    self.training_history.append(snapshot)
                    self._emit(snapshot)

                    if self._step_batch_flag:
                        self._step_batch_flag = False
                        self._pause_event.clear()
                        self.status = "paused"
                        self._emit({
                            "type": "status",
                            "status": "paused",
                            "reason": "step_batch_complete",
                        })

                if self._stop_flag:
                    break

                val_loss, val_acc = self._evaluate(x_val, y_val, loss_fn)

                epoch_summary = {
                    "type": "epoch_update",
                    "epoch": self.current_epoch,
                    "total_epochs": self.total_epochs,
                    "loss": round(float(np.mean(epoch_losses)) if epoch_losses else 0.0, 6),
                    "accuracy": round(float(np.mean(epoch_accs)) if epoch_accs else 0.0, 4),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 4),
                    "timestamp": time.time(),
                }
                self.epoch_history.append(epoch_summary)
                self._emit(epoch_summary)

                if self._step_epoch_flag:
                    self._step_epoch_flag = False
                    self._pause_event.clear()
                    self.status = "paused"
                    self._emit({
                        "type": "status",
                        "status": "paused",
                        "reason": "step_epoch_complete",
                    })

            if self._stop_flag:
                self.status = "idle"
                self._emit({
                    "type": "training_stopped",
                    "epoch": self.current_epoch,
                    "total_snapshots": len(self.training_history),
                })
            else:
                self.status = "completed"
                self._emit({
                    "type": "training_complete",
                    "epochs": self.current_epoch,
                    "total_snapshots": len(self.training_history),
                    "final_accuracy": self.epoch_history[-1]["val_accuracy"] if self.epoch_history else 0,
                })

        except Exception as e:
            self.status = "error"
            self._emit({"type": "training_error", "error": str(e)})
            import traceback

            traceback.print_exc()

    def _extract_activations(self, sample) -> dict:
        """Extract per-layer activations from a single sample for visualization."""
        layer_acts = {}
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                intermediate = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                act = intermediate.predict(sample, verbose=0)
                if len(act.shape) == 2:
                    layer_acts[layer.name] = {
                        "type": "dense",
                        "values": act[0].tolist(),
                    }
                elif len(act.shape) == 4:
                    mean_per_filter = np.mean(act[0], axis=(0, 1)).tolist()
                    layer_acts[layer.name] = {
                        "type": "spatial",
                        "shape": list(act.shape[1:]),
                        "global_mean": float(np.mean(act[0])),
                        "filter_means": mean_per_filter[:16],
                    }
            except Exception:
                pass
        return layer_acts

    def _extract_gradients(self, gradients) -> dict:
        """Extract gradient statistics for visualization."""
        grad_info = {}
        total_norm_sq = 0.0

        for var, grad in zip(self.model.trainable_variables, gradients):
            if grad is None:
                continue
            g = grad.numpy()
            norm = float(np.linalg.norm(g))
            total_norm_sq += norm**2

            grad_info[var.name] = {
                "norm": round(norm, 6),
                "mean": round(float(np.mean(g)), 6),
                "std": round(float(np.std(g)), 6),
                "max_abs": round(float(np.max(np.abs(g))), 6),
                "shape": list(g.shape),
            }

        grad_info["total_norm"] = round(float(np.sqrt(total_norm_sq)), 6)
        return grad_info

    def _extract_weights(self) -> dict:
        """Extract current weight matrices for visualization."""
        weights = {}
        for layer in self.model.layers:
            w = layer.get_weights()
            if len(w) > 0:
                kernel = w[0]
                if len(kernel.shape) == 4:
                    weights[layer.name] = {
                        "type": "conv",
                        "shape": list(kernel.shape),
                        "mean": round(float(np.mean(kernel)), 6),
                        "std": round(float(np.std(kernel)), 6),
                        "norm": round(float(np.linalg.norm(kernel)), 6),
                    }
                else:
                    if kernel.size <= 10000:
                        weights[layer.name] = {
                            "type": "dense",
                            "kernel": kernel.tolist(),
                            "bias": w[1].tolist() if len(w) > 1 else None,
                            "shape": list(kernel.shape),
                        }
                    else:
                        weights[layer.name] = {
                            "type": "dense",
                            "shape": list(kernel.shape),
                            "mean": round(float(np.mean(kernel)), 6),
                            "std": round(float(np.std(kernel)), 6),
                            "norm": round(float(np.linalg.norm(kernel)), 6),
                        }
        return weights

    def _evaluate(self, x_val, y_val, loss_fn) -> tuple:
        """Evaluate model on validation set."""
        predictions = self.model.predict(x_val, verbose=0, batch_size=256)
        val_loss = float(loss_fn(y_val, predictions).numpy())
        pred_classes = np.argmax(predictions, axis=1)
        val_acc = float(np.mean(pred_classes == y_val))
        return val_loss, val_acc

    def _emit(self, message: dict):
        """Send message via WebSocket callback."""
        if self.ws_callback is not None:
            try:
                self.ws_callback(message)
            except Exception:
                pass


training_manager = TrainingManager()
