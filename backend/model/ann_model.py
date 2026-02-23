import tensorflow as tf


def build_ann_model(
    hidden_units: list[int] = [256, 128, 64],
    activation: str = "relu",
    dropout_rate: float = 0.0,
    kernel_initializer: str = "glorot_uniform",
    l2_reg: float = 0.0,
) -> tf.keras.Model:
    """Build a fully-connected ANN for MNIST classification."""
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inputs = tf.keras.Input(shape=(784,), name="input")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=reg,
            name=f"hidden{i+1}",
        )(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout{i+1}")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ann_model")
