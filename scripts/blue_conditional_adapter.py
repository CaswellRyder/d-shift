"""Frozen red probability with a trainable blue/background conditional correction.

p(red) remains the frozen parent's value. The remaining mass is divided by
sigmoid(parent_blue_logit - parent_background_logit + correction). Zero
correction reproduces all parent probabilities. This protects already-accepted
red crops algebraically, not arbitrary downstream selection or localization.
It cannot fix a false red prediction or accept blue when parent p(red) > 0.2
at the fixed 0.8 acceptance threshold. Numerical and export checks are required.
"""
import numpy as np


def conditional_logits(logits, correction):
    """Float64 numerical oracle; logits returned as normalized log probabilities."""
    z, delta = np.asarray(logits, np.float64), np.asarray(correction, np.float64)
    if z.ndim != 2 or z.shape[1] != 3 or delta.shape != (len(z), 1):
        raise ValueError('Require Nx3 logits and Nx1 corrections')
    if not np.isfinite(z).all() or not np.isfinite(delta).all():
        raise ValueError('Require finite logits and corrections')
    total = np.logaddexp.reduce(z, axis=1, keepdims=True)
    nonred = np.logaddexp(z[:, :1], z[:, 2:3]) - total
    blue = z[:, 2:3] - z[:, :1] + delta
    return np.concatenate((nonred - np.logaddexp(0, blue), z[:, 1:2] - total,
                           nonred - np.logaddexp(0, -blue)), axis=1)


def build(parent):
    import keras
    from keras import ops
    if (parent.input_shape != (None, 64, 64, 3) or parent.output_shape != (None, 3)
            or parent.count_params() != 7763 or parent.get_layer('spatial_features').output.shape[-1] != 128):
        raise ValueError('Require original64px three-class7763parameter student')
    frozen = keras.models.clone_model(parent)
    frozen.set_weights(parent.get_weights())
    frozen.trainable = False
    features, z = frozen.get_layer('spatial_features').output, frozen.output
    hidden = keras.layers.Dense(16, activation='relu', name='blue_adapter_hidden')(features)
    delta = keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros', name='blue_adapter_delta')(hidden)
    total = ops.logsumexp(z, axis=(1,), keepdims=True)
    nonred = ops.logsumexp(ops.concatenate((z[:, :1], z[:, 2:3]), axis=1), axis=(1,), keepdims=True) - total
    conditional = z[:, 2:3] - z[:, :1] + delta
    output = ops.concatenate((nonred - ops.softplus(conditional), z[:, 1:2] - total,
                              nonred - ops.softplus(-conditional)), axis=1)
    model = keras.Model(frozen.input, output, name='blue_conditional_adapter')
    if model.count_params() != 9844 or sum(int(np.prod(w.shape)) for w in model.trainable_weights) != 2081:
        raise ValueError('Unexpected trainable or total parameter count')
    return model
