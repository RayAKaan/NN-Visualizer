import tensorflow as tf


def clip_gradients(gradients, clip_norm: float = 5.0):
    grads = [g for g in gradients if g is not None]
    if not grads:
        return gradients, 0.0
    clipped, norm = tf.clip_by_global_norm(grads, clip_norm)
    out = []
    j = 0
    for g in gradients:
        if g is None:
            out.append(None)
        else:
            out.append(clipped[j])
            j += 1
    return out, float(norm.numpy())
