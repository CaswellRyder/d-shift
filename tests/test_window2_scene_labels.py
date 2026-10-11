"""Fail-closed annotation contracts independent of ignored research data."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.window2_scene_labels import scaled_box, validate_partition


def fixture():
    q = dict(frames=[dict(id=5)], samples=[
        dict(id=0, frame_id=5, split='train', source_annotation={'IsGroupOf': '0'}),
        dict(id=1, frame_id=5, split='train', source_annotation={'IsGroupOf': '1'})])
    r = dict(complete_frame_ids=[5], labels={'0': 'red_balloon'},
             excluded_annotations={'1': 'Split group'}, manual_annotations=[
                 dict(id=2, frame_id=5, label='blue_balloon', reason='Visible group member', replaces_group=1)])
    return r, q


def test_valid_partition():
    validate_partition(*fixture())


@pytest.mark.parametrize('case', ['missing', 'overlap', 'source_duplicate', 'frame_duplicate',
    'manual_duplicate', 'manual_collision', 'group_target', 'heldout', 'bad_label', 'wrong_frame', 'bad_group'])
def test_rejects_invalid_review(case):
    r, q = fixture()
    if case == 'missing':
        r['excluded_annotations'] = {}
    elif case == 'overlap':
        r['excluded_annotations']['0'] = 'Conflict'
    elif case == 'source_duplicate':
        q['samples'].append(deepcopy(q['samples'][0]))
    elif case == 'frame_duplicate':
        r['complete_frame_ids'].append(5)
    elif case == 'manual_duplicate':
        r['manual_annotations'].append(deepcopy(r['manual_annotations'][0]))
    elif case == 'manual_collision':
        r['manual_annotations'][0]['id'] = 0
    elif case == 'group_target':
        q['samples'][0]['source_annotation']['IsGroupOf'] = '1'
    elif case == 'heldout':
        q['samples'][0]['split'] = 'test'
    elif case == 'bad_label':
        r['labels']['0'] = 'orange_balloon'
    elif case == 'wrong_frame':
        r['manual_annotations'][0]['frame_id'] = 9
    else:
        r['manual_annotations'][0]['replaces_group'] = 0
    with pytest.raises(ValueError):
        validate_partition(r, q)


def test_original_coordinates_use_independent_axis_scales():
    np.testing.assert_allclose(scaled_box([100, 50, 200, 100], 1000, 500), [32, 24, 64, 48])


@pytest.mark.parametrize('box', [[0, 0, 0, 10], [-1, 0, 3, 4], [0, 0, 321, 240],
    [0, 0, 320, 241], [0, 0, float('nan'), 1], [0, 1, 2], [2, 2, 1, 1]])
def test_bad_coordinates(box):
    with pytest.raises(ValueError):
        scaled_box(box, 320, 240)
