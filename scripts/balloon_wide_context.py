"""Identity-initialized wider spatial context, isolated from runtime defaults."""


def build(parent):
    import keras
    if parent.input_shape != (None, 64, 64, 3) or parent.output_shape != (None, 3):
        raise ValueError('Require 64px three-class parent')
    names = [layer.name for layer in parent.layers]
    expected = ['rgb_0_255', 'normalize', 'conv_8', 'conv_16', 'conv_32',
                'separable_context', 'spatial_pool', 'spatial_features', 'classifier']
    if names != expected or parent.count_params() != 7763:
        raise ValueError('Require frozen separable-context architecture')
    inputs = keras.Input((64, 64, 3), name='rgb_0_255')
    x = inputs
    for layer in parent.layers[1:]:
        clone = type(layer).from_config(layer.get_config())
        x = clone(x)
        clone.set_weights(layer.get_weights())
        if layer.name == 'separable_context':
            residual = keras.layers.DepthwiseConv2D(5, padding='same', use_bias=False,
                depthwise_initializer='zeros', name='wide_context_delta')(x)
            x = keras.layers.Add(name='wide_context_add')([x, residual])
            x = keras.layers.ReLU(name='wide_context_relu')(x)
    model = keras.Model(inputs, x, name='wide_context_student')
    if model.count_params() != 8563:
        raise ValueError('Unexpected parameter count')
    return model


def cost():
    # Three stride-2 convolutions followed by a stride-1 separable convolution.
    stem = 32*32*3*8*9 + 16*16*8*16*9 + 8*8*16*32*9
    separable = 8*8*32*9 + 8*8*32*32
    head = 2*2*32*3
    parent = stem + separable + head
    extra = 8*8*32*25
    return dict(parent_parameters=7763, parameters=8563, added_parameters=800,
        parent_macs=parent, added_macs=extra, macs=parent+extra,
        mac_ratio=(parent+extra)/parent,
        parent_prepool_receptive_field=31, prepool_receptive_field=63,
        caveat='Analytic convolution/dense MACs only; excludes pooling, elementwise work and runtime overhead. Not measured Pi latency.')
