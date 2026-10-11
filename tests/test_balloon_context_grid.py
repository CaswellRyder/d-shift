import numpy as np
import pytest


def test_context_restores_receptive_field_without_new_weights_or_macs():
    pytest.importorskip('tensorflow')
    import keras
    from scripts.balloon_context_grid import network, spatial_cost
    from scripts.balloon_fine_grid import network as fine_network
    from scripts.train_balloon_dense_proposals import network as coarse_network
    keras.utils.set_random_seed(47)
    fine = fine_network()
    keras.utils.set_random_seed(47)
    context = network()
    coarse = coarse_network()
    f, c, old = [spatial_cost(m) for m in (fine, context, coarse)]
    assert f['parameters'] == c['parameters'] == old['parameters'] == 2342
    assert f['multiply_accumulates'] == c['multiply_accumulates'] > old['multiply_accumulates']
    assert c['theoretical_receptive_field_input_pixels'] == old['theoretical_receptive_field_input_pixels'] == 47
    assert f['theoretical_receptive_field_input_pixels'] == 31
    assert f['output_stride'] == c['output_stride'] == 4
    assert old['output_stride'] == 8
    assert fine.output_shape == context.output_shape == (None, 30, 40, 6)
    for a, b in zip(fine.get_weights(), context.get_weights(), strict=True):
        np.testing.assert_array_equal(a, b)
