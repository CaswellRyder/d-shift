import copy

import numpy as np
import pytest

from scripts.window1_priority_data import required_multiplier, matched_inputs


def stats(new, total):
    return dict(gradients={k:dict(negative_gradient_margin_derivatives=v)
                          for k, v in [('new_positive', new), ('total', total)]})


def test_strict_power_of_two_cancellation_bound():
    assert required_multiplier({'a':stats([1, 2], [-8, -2])}) == 16
    assert required_multiplier({'a':stats([1, 2], [1, 2])}) == 1
    assert required_multiplier({'a':stats([1, 2], [-7, -2])}) == 16


@pytest.mark.parametrize('results', [{}, {'a':stats([0, 1], [1, 1])},
                                   {'a':stats([-1, 1], [1, 1])},
                                   {'a':stats([1, 1], [float('nan'), 1])}])
def test_invalid_gradient_evidence_rejected(results):
    with pytest.raises(ValueError):
        required_multiplier(results)


def fixture():
    a = {k:np.zeros(3) for k in ('x', 'y', 'preserve', 'scene_mask', 'probes', 'probe_labels', 'batch_pixels', 'batch_labels')}
    a.update(weights=np.full(2547, 4.), sample_identity=[], scene_ids=[], p={'epochs':80})
    b = copy.deepcopy(a)
    b['weights'][2126:2162] = 64
    return a, b


def test_only_new_positive_weights_change():
    a, b = fixture()
    assert matched_inputs(a, b) == list(range(2126, 2162))


@pytest.mark.parametrize('change', ['pixels', 'other_weight', 'wrong_weight', 'epochs'])
def test_mismatched_trial_rejected(change):
    a, b = fixture()
    if change == 'pixels':
        b['x'][0] = 1
    elif change == 'other_weight':
        b['weights'][0] = 64
    elif change == 'wrong_weight':
        b['weights'][2126] = 16
    else:
        b['p']['epochs'] = 81
    with pytest.raises((ValueError, AssertionError)):
        matched_inputs(a, b)
