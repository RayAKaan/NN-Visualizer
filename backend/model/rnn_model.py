import tensorflow as tf


def build_rnn_model(
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.2,
    activation: str = "tanh",
    recurrent_dropout: float = 0.0,
    bidirectional: bool = False,
) -> tf.keras.Model:
    """Build an LSTM-based RNN for MNIST classification.
    Treats each 28x28 image as 28 timesteps of 28 features."""
    inputs = tf.keras.Input(shape=(28, 28), name="input")
    lstm_layer = tf.keras.layers.LSTM(
        lstm_units,
        activation=activation,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
        name="lstm1",
    )
    if bidirectional:
        x = tf.keras.layers.Bidirectional(lstm_layer, name="bilstm1")(inputs)
    else:
        x = lstm_layer(inputs)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout1")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout2")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="rnn_model")
