import pytest

pytest.importorskip('scipy.special')
from scripts.export_constrained_balloon_head import eligible_names  # noqa: E402
from scripts.evaluate_constrained_balloon_head import development_names  # noqa: E402


def trial():
    return dict(results={'control': {'training_gate_pass': False}, 'admit': {'training_gate_pass': True}},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)


def test_control_can_accompany_passing_admission_without_becoming_candidate():
    assert eligible_names(trial()) == ['control', 'admit']


def test_failed_admission_does_not_export_even_if_control_passes():
    value = trial()
    value['results']['admit']['training_gate_pass'] = False
    value['results']['control']['training_gate_pass'] = True
    with pytest.raises(ValueError):
        eligible_names(value)


def test_training_scene_regression_blocks_development():
    names = ['control', 'admit']
    assert development_names(names, {'admit': {'nonregression': False}}) == []
    assert development_names(names, {'admit': {'nonregression': True}}) == names
    with pytest.raises(ValueError):
        development_names(['admit'], {'admit': {'nonregression': True}})
