from copy import deepcopy

import pytest

from scripts.pi_current_balloon_pipeline import compare


def observation():
    return dict(box=[1., 2., 10., 20.], crop_box=[0, 0, 12, 22], label='blue_balloon',
        scores=[.01, .02, .97], accepted=True, raw_accepted=True, suppressed=False)


def test_small_numerical_drift_requires_identical_decisions_and_crops():
    a, b = observation(), observation()
    b['box'][0] += .001
    b['scores'][0] += .00001
    ds, db = compare([a], [b])
    assert 0 < ds < .001 and 0 < db < .02
    for key, value in [('crop_box', [0, 1, 12, 22]), ('accepted', False), ('label', 'background')]:
        bad = dict(b, **{key: value})
        with pytest.raises(ValueError):
            compare([a], [bad])


def test_reject_nonfinite_large_drift_and_changed_count():
    a = observation()
    for field, value in [('box', [float('nan'), 2, 10, 20]), ('box', [2, 2, 10, 20]),
                         ('scores', [.1, .1, .8])]:
        bad = deepcopy(a)
        bad[field] = value
        with pytest.raises(ValueError):
            compare([a], [bad])
    with pytest.raises(ValueError):
        compare([a], [])
