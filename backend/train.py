from __future__ import annotations

import tensorflow as tf

from model_utils import MODEL_PATH


def main() -> None:
    # ---------------- data ----------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # ---------------- model ----------------
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(784,)),

            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.Dropout(0.15),

            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
            tf.keras.layers.Dropout(0.1),

            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # ---------------- compile ----------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ---------------- callbacks ----------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    # ---------------- train ----------------
    model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    # ---------------- save ----------------
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
