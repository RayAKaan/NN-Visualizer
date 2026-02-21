from __future__ import annotations

import tensorflow as tf


def build_cnn_model(
    conv1_filters: int = 32,
    conv2_filters: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    activation: str = "relu",
    kernel_initializer: str = "glorot_uniform",
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(
        conv1_filters,
        (3, 3),
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="conv2d",
    )(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d")(x)
    x = tf.keras.layers.Conv2D(
        conv2_filters,
        (3, 3),
        padding="same",
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="conv2d_1",
    )(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="dense",
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="dense_1")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_mnist")
