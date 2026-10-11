import numpy as np
import pytest

pytest.importorskip('scipy.optimize')
from scripts.balloon_rebased_constraints import rebase  # noqa: E402


def test_rebase_preserves_absolute_margin_for_both_signs_and_background():
    ref = np.array([[1., 3., 7.]])
    current = np.array([[4., -2., 5.]])
    ids = [(0, 0, 2), (0, 1, 0), (0, -2, 1)]
    bounds = np.array([.4, .8, -.2])
    adjusted = rebase(bounds, ids, ref, current)
    for index, (_, target, rival) in enumerate(ids):
        sign, t = (-1 if target<0 else 1), abs(target)
        desired = bounds[index]+sign*(ref[0,t]-ref[0,rival])
        assert adjusted[index]+sign*(current[0,t]-current[0,rival]) == pytest.approx(desired)
    np.testing.assert_array_equal(rebase(bounds, ids, ref, ref), bounds)
    np.testing.assert_allclose(rebase(bounds, ids, ref+50, current-50), adjusted)


def test_rebase_rejects_malformed_or_nonfinite_inputs():
    for ids in ([(1, 1, 0)], [(0, 3, 0)], [(0, -1, 1)]):
        with pytest.raises(ValueError):
            rebase([1.], ids, np.zeros((1,3)), np.zeros((1,3)))
    with pytest.raises(ValueError):
        rebase([np.nan], [(0,1,0)], np.zeros((1,3)), np.zeros((1,3)))


def test_export_requires_both_admission_seeds_but_keeps_failed_controls():
    from scripts.export_clutter_feature_correction import eligible_names
    names = ['control47', 'control48', 'admit47', 'admit48']
    trial = dict(results={n: dict(training_gate_pass=n.startswith('admit')) for n in names},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    assert eligible_names(trial) == names
    trial['results']['admit48']['training_gate_pass'] = False
    with pytest.raises(ValueError):
        eligible_names(trial)


def test_scene_regression_blocks_development_for_both_seeds():
    from scripts.evaluate_clutter_feature_correction import development_names
    names = ['control47', 'control48', 'admit47', 'admit48']
    summaries = {n: dict(nonregression=n.startswith('admit')) for n in names}
    assert development_names(names, summaries) == names
    summaries['admit48']['nonregression'] = False
    assert development_names(names, summaries) == []
