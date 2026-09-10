"""RNN adapter — keras-io Bidirectional-LSTM IMDB sentiment model.

Source checkpoint: ``keras-io/bidirectional-lstm-imdb`` (Hugging Face).
The repository ships a Keras 2 *SavedModel* plus a ``tokenizer.pickle``.
Modern Keras 3 cannot import that legacy SavedModel (its optimizer object is
incompatible), so this adapter rebuilds the exact architecture in Keras 3 and
copies the genuine float32 checkpoint variables (verified: ~87% IMDB test
accuracy, matching the model card) — no training is performed.

Verified checkpoint layout (from ``variables/variables``):
    Embedding(2000, 30) -> BiLSTM(64, return_sequences=True) ->
    BiLSTM(64) -> Dense(1, sigmoid)
"""
from __future__ import annotations

import os
import pickle
import sys
import types

import numpy as np

from model_registry import store
from model_registry.base import ModelAdapter, ModelLoadError
from model_registry.schema import FAMILY_RNN
from model_registry.util import mnist_words_split

HF_REPO = "keras-io/bidirectional-lstm-imdb"
MAXLEN = 200  # pad_sequences(maxlen=200) as in the training notebook
NUM_WORDS = 2000
LABELS = ["negative", "positive"]


def _ensure_files() -> dict[str, str]:
    """Download the checkpoint pieces into the model cache."""
    out = {
        "tokenizer": store.ensure_downloaded(
            store.hf_url(HF_REPO, "tokenizer.pickle"), "kerasio_imdb_tokenizer.pickle"
        ),
        "variables": store.ensure_downloaded(
            store.hf_url(HF_REPO, "variables/variables.data-00000-of-00001"),
            "kerasio_imdb_variables.data-00000-of-00001",
        ),
        "variables_index": store.ensure_downloaded(
            store.hf_url(HF_REPO, "variables/variables.index"), "kerasio_imdb_variables.index"
        ),
    }
    return out


def _load_tokenizer(pickle_path: str):
    """Load the Keras-2 tokenizer pickle without requiring keras_preprocessing."""
    kp = types.ModuleType("keras_preprocessing")
    kpt = types.ModuleType("keras_preprocessing.text")

    class _Tokenizer:
        def texts_to_sequences(self, texts):
            oov_idx = self.word_index.get(self.oov_token) if self.oov_token else None
            nw = self.num_words
            seqs = []
            for text in texts:
                seq = []
                for word in mnist_words_split(text):
                    if not word:
                        continue
                    idx = self.word_index.get(word)
                    if idx is None:
                        if oov_idx is not None:
                            seq.append(oov_idx)
                        continue
                    if nw and idx >= nw:
                        continue
                    seq.append(idx)
                seqs.append(seq)
            return seqs

    kpt.Tokenizer = _Tokenizer
    kp.text = kpt
    sys.modules.setdefault("keras_preprocessing", kp)
    sys.modules.setdefault("keras_preprocessing.text", kpt)
    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


def _pad_pre(seqs, maxlen: int = MAXLEN) -> np.ndarray:
    """Keras-2 pad_sequences semantics: padding='pre', truncating='pre'."""
    out = []
    for s in seqs:
        s = list(s)
        if len(s) > maxlen:
            s = s[len(s) - maxlen:]
        if len(s) < maxlen:
            s = [0] * (maxlen - len(s)) + s
        out.append(s)
    return np.asarray(out, dtype="int32")


class KerasIoImdbBiLSTMAdapter(ModelAdapter):
    model_id = "rnn-bilstm-imdb-kerasio"

    def __init__(self):
        super().__init__()
        self._model = None

    def _describe(self):
        return {
            "id": self.model_id,
            "name": "BiLSTM IMDB (keras-io)",
            "family": FAMILY_RNN,
            "framework": "TensorFlow / Keras (weights rebuilt from keras-io SavedModel)",
            "source": "Hugging Face — keras-io/bidirectional-lstm-imdb",
            "weights": f"hf://{HF_REPO} (variables + tokenizer.pickle)",
            "dataset": "IMDB movie reviews (binary sentiment)",
            "input_type": "text",
            "input_shape": [200],  # tokenized/padded word ids
            "num_classes": 2,
            "preprocessing": "lowercase -> strip punctuation -> map words (top 2000) -> pad/truncate 200",
            "architecture": "Embedding(2000,30) -> 2x BiLSTM(64) -> Dense(1, sigmoid)",
            "parameter_count": 207_585,
            "description": "Keras official BiLSTM IMDB example checkpoint (10 epochs, ~87% test acc).",
            "license": "Apache-2.0",
            "pretrained": True,
        }

    # ------------------------------------------------------------------
    def _variable_prefix(self, files: dict[str, str]) -> str:
        return files["variables"][: files["variables"].rfind(".data")]

    def load(self):
        import tensorflow as tf

        files = _ensure_files()
        self._tokenizer = _load_tokenizer(files["tokenizer"])

        reader = tf.train.load_checkpoint(self._variable_prefix(files))
        shape = reader.get_variable_to_shape_map()

        def var(key):
            return reader.get_tensor(key)

        def attrs(name):
            return var(f"{name}/.ATTRIBUTES/VARIABLE_VALUE")

        def lstm_triple(num):
            # kernel / recurrent_kernel / bias
            return attrs(f"variables/{num}")

        # ---------- build the Keras-3 model -------------------------------
        ins = tf.keras.Input(shape=(MAXLEN,), dtype="int32")
        emb = tf.keras.layers.Embedding(NUM_WORDS, 30)
        bi1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))
        bi2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        x = emb(ins)
        x = bi1(x)
        x = bi2(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(ins, out)
        model(np.zeros((2, MAXLEN), dtype="int32"))  # force build

        emb.set_weights([attrs("layer_with_weights-0/embeddings")])
        dense_layer = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)][0]
        dense_layer.set_weights([attrs("layer_with_weights-3/kernel"), attrs("layer_with_weights-3/bias")])
        bids = [l for l in model.layers if isinstance(l, tf.keras.layers.Bidirectional)]
        # Verified assignment (no swaps) reproduces ~87% IMDB test accuracy.
        for layer, a, b, c in [
            (bids[0].forward_layer, 1, 2, 3),
            (bids[0].backward_layer, 4, 5, 6),
            (bids[1].forward_layer, 7, 8, 9),
            (bids[1].backward_layer, 10, 11, 12),
        ]:
            layer.set_weights([lstm_triple(a), lstm_triple(b), lstm_triple(c)])

        store.write_provenance(self.model_id, {"repo": HF_REPO, "license": "apache-2.0"})
        self._model = model
        return model

    # ------------------------------------------------------------------
    def predict(self, raw_input):
        text = raw_input.get("text")
        if text is None or not str(text).strip():
            raise ModelLoadError("'text' is required for RNN models")
        if self._model is None or not hasattr(self, "_tokenizer"):
            raise ModelLoadError(f"{self.model_id} is not loaded")
        seqs = self._tokenizer.texts_to_sequences([str(text)])
        x = _pad_pre(seqs)
        logit = float(self._model.predict(x, verbose=0)[0][0])
        p_pos = logit
        p_neg = 1.0 - logit
        probs = [p_neg, p_pos]
        idx = 1 if p_pos >= 0.5 else 0
        return idx, float(probs[idx]), probs, list(LABELS), {"review_len_words": len(str(text).split())}
