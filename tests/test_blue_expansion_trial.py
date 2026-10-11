from copy import deepcopy

import numpy as np
import pytest

from scripts.audit_balloon_blue_expansion import counts
from scripts.train_balloon_blue_expansion import training_gate


def report():
    row = dict(retained_fraction=1., current_argmax_correct=10, parent_argmax_correct=10)
    return dict(retention={'red_balloon': dict(row), 'blue_balloon': dict(row)},
                reviewed_background_correct=22, new_blue_accepted=22)


SETTINGS = dict(minimum_parent_acceptance_retained=.99,
                minimum_reviewed_background_correct=22, minimum_new_blue_accepted=22)


def test_training_gate_requires_new_blue_and_prior_competence():
    passing = report()
    assert training_gate(passing, SETTINGS)
    for field in ('new_blue_accepted', 'reviewed_background_correct'):
        bad = deepcopy(passing)
        bad[field] = 21
        assert not training_gate(bad, SETTINGS)
    for color in ('red_balloon', 'blue_balloon'):
        bad = deepcopy(passing)
        bad['retention'][color]['retained_fraction'] = .98
        assert not training_gate(bad, SETTINGS)
        bad = deepcopy(passing)
        bad['retention'][color]['current_argmax_correct'] = 9
        assert not training_gate(bad, SETTINGS)


def test_correct_is_not_same_as_confidently_accepted():
    logits = np.tile([0., 0., 4.], (23, 1))
    logits[0] = [0., 0., .1]
    logits[1] = [4., 0., 0.]
    assert counts(logits) == (22, 21)


@pytest.mark.parametrize('values', [np.zeros((22, 3)), np.full((23, 3), np.nan),
                                   np.full((23, 3), np.inf)])
def test_probe_metrics_reject_invalid_arrays(values):
    with pytest.raises(ValueError):
        counts(values)
