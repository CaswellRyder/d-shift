import pytest

pytest.importorskip('scipy.optimize')
from scripts.export_clutter_round3_head import eligible_names  # noqa: E402
from scripts.evaluate_clutter_round3_head import development_names  # noqa: E402


def test_failed_control_can_accompany_passing_new_negative_fit():
    trial = dict(results={'control': {'training_gate_pass': False}, 'admit': {'training_gate_pass': True}},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    assert eligible_names(trial) == ['control', 'admit']
    trial['results']['admit']['training_gate_pass'] = False
    with pytest.raises(ValueError):
        eligible_names(trial)


def test_no_development_on_training_scene_regression():
    assert development_names(['control', 'admit'], {'admit': {'nonregression': False}}) == []
    assert development_names(['control', 'admit'], {'admit': {'nonregression': True}}) == ['control', 'admit']
