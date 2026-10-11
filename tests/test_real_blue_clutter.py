import numpy as np
import pytest

from scripts.train_real_blue_clutter import calculate, run_path
from scripts.blue_false_alarm_data import data


def arrays():
    logits = np.array([[0., 5., 0.], [0., 0., 5.]])
    return dict(before=logits, after=logits.copy(), labels=np.array([1, 2]), ordinary=np.array([True, True]),
                blue_after=np.tile([0., 0., 5.], (23, 1)),
                clutter_after=np.tile([5., 0., 0.], (22, 1)),
                real_after=np.tile([5., 0., 0.], (39, 1)))


def test_real_background_gate_does_not_replace_positive_retention():
    a = arrays()
    assert calculate(a)['training_gate_pass']
    a['after'][1] = [5., 0., 0.]
    assert not calculate(a)['training_gate_pass']


def test_real_and_old_background_checks_are_separate():
    a = arrays()
    a['real_after'][:3] = [0., 0., 5.]
    assert not calculate(a)['training_gate_pass']
    a = arrays()
    a['clutter_after'][0] = [0., 0., 5.]
    assert not calculate(a)['training_gate_pass']


def test_new_blue_confidence_still_required():
    a = arrays()
    a['blue_after'][:2] = [0., 0., .1]
    assert not calculate(a)['training_gate_pass']


@pytest.mark.parametrize('field', ['blue_after', 'clutter_after', 'real_after'])
def test_invalid_probes_rejected(field):
    a = arrays()
    a[field][0, 0] = np.nan
    with pytest.raises(ValueError):
        calculate(a)


def test_unknown_arm_rejected_before_data_reads():
    with pytest.raises(ValueError):
        data('best')
    with pytest.raises(ValueError):
        run_path('admit', 49)
