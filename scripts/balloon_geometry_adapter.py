"""Zero-initialized box-only residual branch; parent scores/features stay frozen."""
import hashlib

import numpy as np


def network(parent):
    import keras
    layers = keras.layers
    if parent.input_shape != (None, 120, 160, 3) or parent.output_shape != (None, 15, 20, 6):
        raise ValueError('Require coarse proposal parent')
    for layer in parent.layers:
        layer.trainable = False
    geometry = parent.get_layer('center_geometry')
    features = geometry.input
    original = layers.Conv2D(4, 1, name='frozen_geometry_logits', trainable=False)
    logits = original(features)
    original.set_weights(geometry.get_weights())
    residual = layers.DepthwiseConv2D(3, padding='same', use_bias=False,
                                    name='box_residual_depthwise')(features)
    residual = layers.Conv2D(16, 1, activation='relu', name='box_residual_hidden')(residual)
    residual = layers.Conv2D(4, 1, kernel_initializer='zeros', bias_initializer='zeros',
                            name='box_residual_delta')(residual)
    refined = layers.Activation('sigmoid', name='refined_geometry')(
        layers.Add(name='geometry_residual_add')([logits, residual]))
    output = layers.Concatenate(name='adapted_proposal_output')([parent.get_layer('center_logits').output, refined])
    model = keras.Model(parent.input, output)
    if model.count_params() != 3026 or sum(int(np.prod(w.shape)) for w in model.trainable_weights) != 684:
        raise ValueError('Unexpected geometry branch capacity')
    return model


def frozen_weights(model):
    result = {}
    for layer in model.layers:
        if not layer.trainable and layer.weights:
            result[layer.name] = [hashlib.sha256(np.ascontiguousarray(w.numpy()).tobytes()).hexdigest()
                                  for w in layer.weights]
    return result


def geometry_loss(target, prediction):
    import tensorflow as tf
    from scripts.balloon_dense_iou_loss import overlap_penalty
    mask = target[..., 8:9]
    l1 = tf.reduce_sum(tf.abs(prediction[..., 2:] - target[..., 4:8]) * mask)
    l1 /= tf.maximum(4 * tf.reduce_sum(mask), 1.)
    return 5 * l1 + 2 * overlap_penalty(target, prediction)
