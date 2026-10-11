import pytest

pytest.importorskip('scipy.special')
from scripts.export_scene_constrained_balloon_head import eligible_names  # noqa: E402
from scripts.evaluate_scene_constrained_balloon_head import development_names  # noqa: E402


def test_scene_trial_export_requires_admission_not_control_success():
    value = dict(results={'control': {'training_gate_pass': False}, 'admit': {'training_gate_pass': True}},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    assert eligible_names(value) == ['control', 'admit']
    value['results']['admit']['training_gate_pass'] = False
    with pytest.raises(ValueError):
        eligible_names(value)


def test_scene_trial_development_requires_actual_full_scene_nonregression():
    names = ['control', 'admit']
    assert development_names(names, {'admit': {'nonregression': False}}) == []
    assert development_names(names, {'admit': {'nonregression': True}}) == names
    with pytest.raises(ValueError):
        development_names(['admit'], {'admit': {'nonregression': True}})
