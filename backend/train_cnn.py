from __future__ import annotations

import os

import tensorflow as tf

from model.cnn_model import build_cnn_model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = build_cnn_model()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
)

save_path = os.path.join(os.path.dirname(__file__), "nn_visual_cnn.keras")
model.save(save_path)
print(f"Saved CNN model to {save_path}")

h5_path = os.path.join(os.path.dirname(__file__), "nn_visual_cnn.h5")
model.save(h5_path)
print(f"Saved CNN model to {h5_path}")
