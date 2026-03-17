from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def _prepare_data(samples: np.ndarray) -> np.ndarray:
    x = samples.astype(np.float32)
    x = x.reshape((x.shape[0], -1))
    return x


def train_autoencoder(x: np.ndarray, latent_dim: int = 16, epochs: int = 5, batch_size: int = 32) -> Tuple[Tuple[object, object], Dict]:
    import tensorflow as tf

    x = _prepare_data(x)
    input_dim = x.shape[1]
    inputs = tf.keras.Input(shape=(input_dim,))
    h = tf.keras.layers.Dense(64, activation="relu")(inputs)
    z = tf.keras.layers.Dense(latent_dim, activation="linear")(h)
    h2 = tf.keras.layers.Dense(64, activation="relu")(z)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(h2)
    model = tf.keras.Model(inputs, outputs)
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    dh = tf.keras.layers.Dense(64, activation="relu")(decoder_input)
    decoder_output = tf.keras.layers.Dense(input_dim, activation="sigmoid")(dh)
    decoder = tf.keras.Model(decoder_input, decoder_output)
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=0)
    return (model, decoder), {"loss": float(history.history["loss"][-1])}


def train_vae(x: np.ndarray, latent_dim: int = 8, epochs: int = 5, batch_size: int = 32) -> Tuple[Tuple[object, object], Dict]:
    import tensorflow as tf

    x = _prepare_data(x)
    input_dim = x.shape[1]
    inputs = tf.keras.Input(shape=(input_dim,))
    h = tf.keras.layers.Dense(64, activation="relu")(inputs)
    z_mean = tf.keras.layers.Dense(latent_dim)(h)
    z_log_var = tf.keras.layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

    z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])
    h2 = tf.keras.layers.Dense(64, activation="relu")(z)
    outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(h2)

    vae = tf.keras.Model(inputs, outputs)

    recon_loss = tf.keras.losses.mse(inputs, outputs)
    recon_loss = tf.reduce_mean(recon_loss) * input_dim
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae.add_loss(recon_loss + kl_loss)
    vae.compile(optimizer="adam")
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    dh = tf.keras.layers.Dense(64, activation="relu")(decoder_input)
    decoder_output = tf.keras.layers.Dense(input_dim, activation="sigmoid")(dh)
    decoder = tf.keras.Model(decoder_input, decoder_output)

    vae.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=0)
    return (vae, decoder), {"recon_loss": float(recon_loss.numpy()), "kl_loss": float(kl_loss.numpy())}


def train_gan(x: np.ndarray, latent_dim: int = 16, epochs: int = 5, batch_size: int = 32) -> Tuple[Tuple[object, object], Dict]:
    import tensorflow as tf

    x = _prepare_data(x)
    input_dim = x.shape[1]

    generator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(input_dim, activation="sigmoid"),
    ])

    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    discriminator.trainable = False
    z = tf.keras.Input(shape=(latent_dim,))
    fake = generator(z)
    validity = discriminator(fake)
    combined = tf.keras.Model(z, validity)
    combined.compile(optimizer="adam", loss="binary_crossentropy")

    for _ in range(epochs):
        idx = np.random.randint(0, x.shape[0], batch_size)
        real = x[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim)).astype(np.float32)
        fake = generator.predict(noise, verbose=0)
        discriminator.trainable = True
        discriminator.train_on_batch(real, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake, np.zeros((batch_size, 1)))
        discriminator.trainable = False
        combined.train_on_batch(noise, np.ones((batch_size, 1)))

    return (generator, discriminator), {"status": "trained"}
