import numpy as np
import pytest

from scripts.balloon_blue_final_fit import metrics, run_path
from scripts.summarize_balloon_mask_convergence import meets_target


def arrays():
    logits = np.array([[0., 5., 0.], [0., 0., 5.]])
    return dict(before=logits, start=logits, after=logits.copy(),
                labels=np.array([1, 2]), ordinary=np.array([True, True]),
                blue_after=np.tile([0., 0., 5.], (23, 1)),
                probe_after=np.tile([5., 0., 0.], (22, 1)))


def test_metrics_recompute_both_positive_retention_and_clutter():
    a = arrays()
    result = metrics(a)
    assert result['new_blue_correct'] == 23
    assert result['new_blue_accepted'] == 23
    assert result['training_gate_pass']
    a['probe_after'][0] = [0., 0., 5.]
    assert not metrics(a)['training_gate_pass']
    a = arrays()
    a['after'][0] = [5., 0., 0.]
    assert not metrics(a)['training_gate_pass']


@pytest.mark.parametrize('probe', [np.zeros((21, 3)), np.full((22, 3), np.nan)])
def test_bad_clutter_probes_fail_closed(probe):
    a = arrays()
    a['probe_after'] = probe
    with pytest.raises(ValueError):
        metrics(a)


@pytest.mark.parametrize('mode,seed', [('best', 47), ('admit', 49)])
def test_undeclared_checkpoint_selection_rejected(mode, seed):
    with pytest.raises(ValueError):
        run_path(mode, seed)


def test_development_target_requires_both_metrics_both_colors():
    good = {c: dict(precision=.94, recall=.94) for c in ('red_balloon', 'blue_balloon')}
    assert meets_target(good)
    for color in good:
        for key in ('precision', 'recall'):
            for value in (.939, None, float('nan')):
                candidate = {c: dict(r) for c, r in good.items()}
                candidate[color][key] = value
                assert not meets_target(candidate)
