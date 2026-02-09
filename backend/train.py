from __future__ import annotations

import tensorflow as tf
from model_utils import MODEL_PATH


def main() -> None:
    # ===============================
    # Load MNIST
    # ===============================
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # IMPORTANT: flatten for visualizer
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # ===============================
    # Model (Optimized Dense)
    # ===============================
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(784,)),

            # ---- Hidden layer 1 ----
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),

            # ---- Hidden layer 2 ----
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_initializer="he_normal",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.1),

            # ---- Output ----
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # ===============================
    # Optimizer + LR schedule
    # ===============================
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=4000,
        decay_rate=0.96,
        staircase=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ===============================
    # Callbacks
    # ===============================
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    # ===============================
    # Train
    # ===============================
    model.fit(
        x_train,
        y_train,
        epochs=40,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    # ===============================
    # Save
    # ===============================
    model.save(MODEL_PATH)
    print(f"Saved optimized Dense model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
