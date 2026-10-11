"""Joint center/box residual using the same tiny spatial refinement features."""
import numpy as np

from scripts.balloon_geometry_adapter import network as geometry_network, frozen_weights


def network(parent):
    import keras
    geometry = geometry_network(parent)
    hidden = geometry.get_layer('box_residual_hidden').output
    delta = keras.layers.Conv2D(2, 1, kernel_initializer='zeros', bias_initializer='zeros',
                                name='center_residual_delta')(hidden)
    heat = keras.layers.Add(name='refined_center_logits')([geometry.get_layer('center_logits').output, delta])
    output = keras.layers.Concatenate(name='center_geometry_output')([heat, geometry.get_layer('refined_geometry').output])
    model = keras.Model(geometry.input, output)
    if model.count_params() != 3060 or sum(int(np.prod(w.shape)) for w in model.trainable_weights) != 718:
        raise ValueError('Unexpected joint refinement capacity')
    return model


def joint_loss(target, prediction):
    from scripts.balloon_dense_iou_loss import giou_loss
    return giou_loss(target, prediction)


__all__ = ['network', 'joint_loss', 'frozen_weights']
