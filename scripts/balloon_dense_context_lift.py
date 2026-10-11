"""Lift the trained separable context to dense convolution without a cold start.

The initial function is algebraically equivalent (up to floating-point order).
Fine-tuning can then learn different spatial filters for each output channel.
This is a research alternative, not a runtime default or a speed claim.
"""
import numpy as np


def compose_kernel(depthwise, pointwise):
    depthwise, pointwise = np.asarray(depthwise), np.asarray(pointwise)
    if (depthwise.shape != (3, 3, 32, 1) or pointwise.shape != (1, 1, 32, 32)
            or not np.isfinite(depthwise).all() or not np.isfinite(pointwise).all()):
        raise ValueError('Require finite32-channel multiplier1 separable weights')
    return depthwise[:, :, :, 0, None] * pointwise[0, 0][None, None, :, :]


def build(parent):
    import keras
    expected = ['rgb_0_255', 'normalize', 'conv_8', 'conv_16', 'conv_32',
                'separable_context', 'spatial_pool', 'spatial_features', 'classifier']
    if (parent.input_shape != (None, 64, 64, 3) or parent.output_shape != (None, 3)
            or [r.name for r in parent.layers] != expected or parent.count_params() != 7763):
        raise ValueError('Require frozen64px three-class separable-context student')
    source = parent.get_layer('separable_context')
    cfg = source.get_config()
    if (not isinstance(source, keras.layers.SeparableConv2D) or cfg['activation'] != 'relu'
            or tuple(cfg['strides']) != (1, 1) or tuple(cfg['dilation_rate']) != (1, 1)
            or cfg['padding'] != 'same' or cfg['data_format'] != 'channels_last'
            or cfg['depth_multiplier'] != 1 or not cfg['use_bias']):
        raise ValueError('Unsupported context operator')
    depthwise, pointwise, bias = source.get_weights()
    kernel = compose_kernel(depthwise, pointwise)
    inputs = keras.Input((64, 64, 3), name='rgb_0_255')
    x = inputs
    for old in parent.layers[1:]:
        if old.name == 'separable_context':
            layer = keras.layers.Conv2D(32, 3, padding='same', activation='relu', name='context')
            x = layer(x)
            layer.set_weights([kernel, bias])
        else:
            layer = type(old).from_config(old.get_config())
            x = layer(x)
            layer.set_weights(old.get_weights())
    model = keras.Model(inputs, x, name='dense_context_lift_student')
    if model.count_params() != 15667:
        raise ValueError('Unexpected dense-context parameter count')
    return model


def cost():
    stem = 32*32*3*8*9 + 16*16*8*16*9 + 8*8*16*32*9
    head = 2*2*32*3
    separable = 8*8*32*9 + 8*8*32*32
    dense = 8*8*32*32*9
    parent = stem + separable + head
    return dict(parent_parameters=7763, parameters=15667, added_parameters=7904,
        parent_macs=parent, macs=stem+dense+head, added_macs=dense-separable,
        mac_ratio=(stem+dense+head)/parent, prepool_receptive_field=31,
        scope='Analytic convolution/dense MACs only. No measured Pi latency, frame-rate or accuracy improvement.')
