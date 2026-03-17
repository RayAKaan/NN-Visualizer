from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .models import train_autoencoder, train_vae, train_gan
from ..visualization.rendering import render_gray


class GenerativeManager:
    def __init__(self) -> None:
        self.autoencoder = None
        self.auto_decoder = None
        self.vae = None
        self.vae_decoder = None
        self.gan = None
        self.input_dim = None

    def train(self, x: np.ndarray, mode: str, epochs: int = 5) -> Dict:
        mode = (mode or "vae").lower()
        self.input_dim = x.reshape((x.shape[0], -1)).shape[1]
        if mode == "ae":
            (self.autoencoder, self.auto_decoder), metrics = train_autoencoder(x, epochs=epochs)
        elif mode == "gan":
            self.gan, metrics = train_gan(x, epochs=epochs)
        else:
            (self.vae, self.vae_decoder), metrics = train_vae(x, epochs=epochs)
        return {"mode": mode, "metrics": metrics}

    def sample(self, mode: str, n_samples: int, size: int) -> Dict:
        mode = (mode or "vae").lower()
        samples = []
        if mode == "ae" and self.auto_decoder is not None:
            noise = np.random.normal(0, 1, (n_samples, 16)).astype(np.float32)
            recon = self.auto_decoder.predict(noise, verbose=0)
            for i in range(n_samples):
                img = recon[i].reshape(size, size)
                samples.append(render_gray(img))
        elif mode == "gan" and self.gan is not None:
            generator, _ = self.gan
            z = np.random.normal(0, 1, (n_samples, 16)).astype(np.float32)
            fake = generator.predict(z, verbose=0)
            for i in range(n_samples):
                img = fake[i].reshape(size, size)
                samples.append(render_gray(img))
        elif mode == "vae" and self.vae_decoder is not None:
            z = np.random.normal(0, 1, (n_samples, 8)).astype(np.float32)
            fake = self.vae_decoder.predict(z, verbose=0)
            for i in range(n_samples):
                img = fake[i].reshape(size, size)
                samples.append(render_gray(img))
        return {"mode": mode, "samples": samples}


generative_manager = GenerativeManager()
