from __future__ import annotations

import tensorflow as tf


def build_ann(activation: str = "relu", initializer: str = "glorot_uniform", weight_decay: float = 0.0, dropout: float = 0.0) -> tf.keras.Model:
    activation_layer = tf.keras.layers.LeakyReLU() if activation == "leaky_relu" else activation
    layers = [
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(
            128,
            activation=activation_layer,
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        ),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(
        tf.keras.layers.Dense(
            64,
            activation=activation_layer,
            kernel_initializer=initializer,
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
        )
    )
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(10, activation="softmax", kernel_initializer=initializer))
    return tf.keras.Sequential(layers)
