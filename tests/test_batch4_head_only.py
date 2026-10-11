from types import SimpleNamespace

import numpy as np
import pytest

from dtr.data import read_json
from scripts.train_batch4_head_only import PROTOCOL, fold_head, frozen_weights


def test_matched_data_and_schedule_only_trainable_scope_changes():
    full = read_json('configs/batch4-retraining-20261011.json')
    head = read_json(PROTOCOL)
    assert {k: head[k] for k in full if k != 'recipe'} == {k: v for k, v in full.items() if k != 'recipe'}
    assert head['trainable_parameters'] == 387
    assert head['frozen_parameters'] + head['trainable_parameters'] == head['parameters'] == 7763


def model():
    layers = []
    for name in ('feature', 'classifier'):
        weights = [np.array([1., 2.])]
        layer = SimpleNamespace(name=name, values=weights)
        layer.get_weights = lambda layer=layer: [v.copy() for v in layer.values]
        layer.set_weights = lambda values, layer=layer: setattr(layer, 'values', [v.copy() for v in values])
        layers.append(layer)
    return SimpleNamespace(layers=layers, get_layer=lambda n: next(layer for layer in layers if layer.name == n))


def test_fold_updates_only_classifier_and_rejects_backbone_drift():
    pupil, head = model(), model()
    before = frozen_weights(pupil)
    head.get_layer('classifier').set_weights([np.array([3., 4.])])
    fold_head(pupil, head, before)
    np.testing.assert_array_equal(pupil.get_layer('classifier').get_weights()[0], [3., 4.])
    np.testing.assert_array_equal(frozen_weights(pupil)[0], before[0])
    pupil.get_layer('feature').set_weights([np.array([5., 6.])])
    with pytest.raises(ValueError, match='Frozen feature'):
        fold_head(pupil, head, before)
