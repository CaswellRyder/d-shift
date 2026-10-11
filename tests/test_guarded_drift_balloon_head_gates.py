import pytest

pytest.importorskip('scipy.optimize')
from scripts.export_guarded_drift_balloon_head import eligible_names  # noqa: E402
from scripts.evaluate_guarded_drift_balloon_head import development_names  # noqa: E402


def test_guarded_drift_export_rejects_unqualified_training():
    trial = dict(results={'control': {'training_gate_pass': False}, 'admit': {'training_gate_pass': True}},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    assert eligible_names(trial) == ['control', 'admit']
    trial['eligible_for_full_scene_comparison'] = False
    with pytest.raises(ValueError):
        eligible_names(trial)


def test_guarded_drift_scene_regression_blocks_development():
    names = ['control', 'admit']
    assert development_names(names, {'admit': {'nonregression': False}}) == []
    assert development_names(names, {'admit': {'nonregression': True}}) == names
