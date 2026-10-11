import pytest

pytest.importorskip('scipy.optimize')
from scripts.export_background_constrained_balloon_head import eligible_names  # noqa: E402
from scripts.evaluate_background_constrained_balloon_head import development_names  # noqa: E402


def test_background_admission_must_actually_correct_backgrounds():
    trial = dict(results={'control': {'training_gate_pass': True},
                         'admit': {'training_gate_pass': True, 'all_background_correct': True}},
        eligible_for_full_scene_comparison=True, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False)
    assert eligible_names(trial) == ['control', 'admit']
    trial['results']['admit']['all_background_correct'] = False
    with pytest.raises(ValueError):
        eligible_names(trial)


def test_control_cannot_unlock_development_when_treatment_regresses():
    names = ['control', 'admit']
    assert development_names(names, {'control': {'nonregression': True}, 'admit': {'nonregression': False}}) == []
