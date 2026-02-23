from __future__ import annotations

import tensorflow as tf


def clip_gradients(gradients: list[tf.Tensor | None], clip_value: float = 5.0) -> list[tf.Tensor | None]:
    return [tf.clip_by_value(g, -clip_value, clip_value) if g is not None else g for g in gradients]
