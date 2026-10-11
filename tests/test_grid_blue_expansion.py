import pytest
from copy import deepcopy
import numpy as np

from scripts.grid_blue_expansion_data import combine, expected
from scripts.balloon_dense_public_data import encode_public
from scripts.audit_grid_blue_expansion import gate
from scripts.diagnose_grid_blue_training import center_probe


def test_sparse_annotations_join_by_original_source_not_crop():
    positive = dict(source_image='TRAIN/a', id=1, split='train', label='blue_balloon', box=[40, 40, 60, 60])
    negative = dict(source_image='TRAIN/a', id=2, split='train', label='background', crop_box=[100, 100, 140, 140])
    groups = combine([positive], [negative])
    assert len(groups) == 1
    group = groups['TRAIN/a']
    target, regions, skipped, _ = encode_public(group['truth'], group['backgrounds'])
    assert regions == [negative['crop_box']]
    assert not skipped
    assert target[..., 8].sum() == 1
    # Unknown image corners are not silently labeled background.
    assert target[0, 0, 2:4].sum() == 0


def test_small_background_region_does_not_expand_to_an_unreviewed_cell():
    target, _, _, _ = encode_public([], [[1, 1, 5, 5]])
    assert target[..., 2:4].sum() == 0


@pytest.mark.parametrize('split', ['test', 'validation'])
def test_nontrain_rows_rejected(split):
    row = dict(source_image='a', id=0, split=split, label='blue_balloon', box=[1, 1, 5, 5])
    with pytest.raises(ValueError, match='TRAIN'):
        combine([row], [])


def test_unknown_arm_rejected_before_data_access():
    with pytest.raises(ValueError, match='arm'):
        expected('best')


def test_gate_requires_new_coverage_old_coverage_and_negative_rejection():
    before = {split: {color: dict(targets=10, learned=7) for color in ('red_balloon', 'blue_balloon')}
              for split in ('base_fit', 'base_holdout', 'extra_fit')}
    after = deepcopy(before)
    after['extra_fit']['blue_balloon']['learned'] = 8
    assert gate(before, after, .8, .6)
    assert not gate(before, after, .8, .8)
    after['base_fit']['red_balloon']['learned'] = 6
    assert not gate(before, after, .8, .6)
    after['extra_fit']['blue_balloon']['targets'] = 11
    with pytest.raises(ValueError, match='target count'):
        gate(before, after, .8, .6)


def test_grouped_holdout_is_not_a_selection_signal():
    before = {split: {color: dict(targets=10, learned=7) for color in ('red_balloon', 'blue_balloon')}
              for split in ('base_fit', 'base_holdout', 'extra_fit')}
    after = deepcopy(before)
    after['extra_fit']['blue_balloon']['learned'] = 8
    after['base_holdout']['blue_balloon']['learned'] = 0
    assert gate(before, after, .8, .6)


def test_center_geometry_and_peak_are_distinct_diagnostics():
    value = np.zeros((15, 20, 6), np.float32)
    value[..., 1] = -8
    value[2, 3] = [0, 2, .5, .5, 16 / 320, 16 / 240]
    found = center_probe(value, [48, 32, 64, 48])
    assert found['cell'] == [2, 3]
    assert found['iou'] == pytest.approx(1)
    assert found['heat'] > .8
    assert found['local_peak']
    value[2, 4, 1] = 3
    assert not center_probe(value, [48, 32, 64, 48])['local_peak']
