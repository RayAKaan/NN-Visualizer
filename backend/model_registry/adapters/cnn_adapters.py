"""Adapters for ImageNet CNN models from Keras Applications.

All four models use genuine ``weights="imagenet"`` checkpoints managed by
Keras itself (cached in ``~/.keras`` / ``KERAS_HOME`` on first load — lazy,
never at server startup).  MobileNetV2 / MobileNetV3Small are the preferred
lightweight CPU models; ResNet50 / VGG16 are heavier reference architectures.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from model_registry.base import ModelAdapter, ModelLoadError
from model_registry.schema import FAMILY_CNN
from model_registry.util import load_image_from_base64

SIZE = (224, 224)


class KerasImageNetCNNAdapter(ModelAdapter):
    """Generic adapter around tf.keras.applications models."""

    # -- subclass overrides -------------------------------------------------
    model_id = ""
    display_name = ""
    keras_app_name = ""          # e.g. "MobileNetV2"
    preprocessing_module = None  # module with preprocess_input
    arch_desc = ""
    params = 0
    desc = ""
    preferred = False

    def __init__(self):
        self._model = None
        super().__init__()

    def _describe(self):
        return {
            "id": self.model_id,
            "name": self.display_name,
            "family": FAMILY_CNN,
            "framework": "TensorFlow / Keras Applications",
            "source": "Keras Applications — weights='imagenet'",
            "weights": "imagenet (Keras cache, auto-downloaded on first load)",
            "dataset": "ImageNet-1k (1000 classes)",
            "input_type": "image",
            "input_shape": [224, 224, 3],
            "num_classes": 1000,
            "preprocessing": "RGB image -> resize 224x224 -> model-specific preprocess_input",
            "architecture": self.arch_desc,
            "parameter_count": self.params,
            "description": self.desc,
            "license": "Apache-2.0",
            "pretrained": True,
        }

    def load(self):
        ctor = getattr(tf.keras.applications, self.keras_app_name)
        self._model = ctor(weights="imagenet")
        return self._model

    def predict(self, raw_input):
        if "image" not in raw_input:
            raise ModelLoadError("'image' (base64) is required for CNN models")
        if self._model is None:
            raise ModelLoadError(f"{self.model_id} is not loaded")
        img_uint8 = (load_image_from_base64(raw_input["image"], size=SIZE) * 255.0).astype(np.uint8)
        x = self.preprocessing_module.preprocess_input(np.expand_dims(img_uint8, 0))
        logits = self._model.predict(x, verbose=0)[0]
        probs = np.asarray(logits, dtype=np.float64)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        idx = int(np.argmax(probs))
        labels = _imagenet_labels_for(probs)
        return idx, float(probs[idx]), probs.tolist(), labels, None


_IMAGENET_LABELS_CACHE = None


def _imagenet_labels_for(probs):
    """Build a 1000-element class-name array via keras decode_predictions."""
    global _IMAGENET_LABELS_CACHE
    if _IMAGENET_LABELS_CACHE is not None:
        return _IMAGENET_LABELS_CACHE or None
    try:
        import json

        fpath = tf.keras.utils.get_file(
            "imagenet_class_index.json",
            "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json",
            cache_subdir="models",
            file_hash="c2c37ea517e94d9795004a39431a14cb",
        )
        with open(fpath, encoding="utf-8") as fh:
            class_index = json.load(fh)  # {"0": [wnid, "tench"], ...}
        labels = [""] * 1000
        for index_key, (_wnid, name) in class_index.items():
            labels[int(index_key)] = name
        _IMAGENET_LABELS_CACHE = labels
        return labels
    except Exception:
        _IMAGENET_LABELS_CACHE = []
        return None


class MobileNetV2Adapter(KerasImageNetCNNAdapter):
    model_id = "cnn-mobilenetv2"
    display_name = "MobileNetV2 (ImageNet)"
    keras_app_name = "MobileNetV2"
    preprocessing_module = tf.keras.applications.mobilenet_v2
    arch_desc = "Inverted-residual CNN, depthwise-separable blocks"
    params = 3_538_984
    desc = "Lightweight efficient CNN (~3.5M params). Great CPU default."
    preferred = True


class MobileNetV3SmallAdapter(KerasImageNetCNNAdapter):
    model_id = "cnn-mobilenetv3small"
    display_name = "MobileNetV3-Small (ImageNet)"
    keras_app_name = "MobileNetV3Small"
    preprocessing_module = tf.keras.applications.mobilenet_v3
    arch_desc = "Hard-swish MobileNetV3 small variant"
    params = 2_554_968
    desc = "Smallest / fastest ImageNet CNN (~2.5M params). Best CPU pick."
    preferred = True


class ResNet50Adapter(KerasImageNetCNNAdapter):
    model_id = "cnn-resnet50"
    display_name = "ResNet50 (ImageNet)"
    keras_app_name = "ResNet50"
    preprocessing_module = tf.keras.applications.resnet50
    arch_desc = "Residual network with 50 layers (bottleneck blocks)"
    params = 25_636_712
    desc = "Classic academic reference CNN (~25.6M params). Slower on CPU."
    preferred = False


class VGG16Adapter(KerasImageNetCNNAdapter):
    model_id = "cnn-vgg16"
    display_name = "VGG16 (ImageNet)"
    keras_app_name = "VGG16"
    preprocessing_module = tf.keras.applications.vgg16
    arch_desc = "Plain 16-layer CNN, 3x3 conv stacks"
    params = 138_357_544
    desc = "Classic reference CNN (~138M params). Requires ~3-4 GB RAM to load."
    preferred = False
