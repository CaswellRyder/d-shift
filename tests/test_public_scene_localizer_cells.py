import json

import numpy as np

from scripts.diagnose_public_scene_localizer_cells import inspect


TRUTH = [dict(label='red_balloon', box=[24, 24, 40, 40])]


def values():
    v = np.full((15, 20, 6), .5, np.float32)
    v[..., :2] = -10
    return v


def test_geometry_failure_before_scores():
    row = inspect(values(), TRUTH)[0]
    assert row['first_loss'] == 'all_cells'


def test_confidence_failure_with_correct_box():
    v = values()
    v[2, 2, 2:] = [0, 0, 16/320, 16/240]
    row = inspect(v, TRUTH)[0]
    assert row['first_loss'] == 'confident_cells'
    assert row['center_box_iou'] > .99


def test_peak_filter_failure():
    v = values()
    v[2, 2, 2:] = [0, 0, 16/320, 16/240]
    v[2, 2, 0], v[2, 1, 0] = 3, 5
    assert inspect(v, TRUTH)[0]['first_loss'] == 'local_peaks'


def test_successful_decode():
    v = values()
    v[2, 2, 2:] = [0, 0, 16/320, 16/240]
    v[2, 2, 0] = 5
    row = inspect(v, TRUTH)[0]
    assert row['first_loss'] == 'localized'
    assert row['center_cell'] == [2, 2]
    json.dumps(row, allow_nan=False)
