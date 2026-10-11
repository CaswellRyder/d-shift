import pytest

from scripts.evaluate_batch4_retraining import development_names
from scripts.export_batch4_retraining import eligible_names


def trial():
    return dict(results={f'{arm}{seed}': dict(training_gate_pass=arm == 'admit')
                         for arm in ('control', 'admit') for seed in (47, 48)},
        eligible_for_full_scene_comparison=True, development_evaluated=False,
        test_evaluated=False, pi_tested=False, deployment_approved=False, qualification_achieved=False)


def test_controls_are_comparators_not_required_to_fit_new_crops():
    names = eligible_names(trial())
    assert names == ['control47', 'control48', 'admit47', 'admit48']
    summaries = {n: dict(nonregression=n.startswith('admit')) for n in names}
    assert development_names(names, summaries) == names
    summaries['admit48']['nonregression'] = False
    assert development_names(names, summaries) == []


@pytest.mark.parametrize('case', ['failed_seed', 'missing_control', 'scope', 'forged_gate'])
def test_export_fails_closed(case):
    t = trial()
    if case == 'failed_seed':
        t['results']['admit47']['training_gate_pass'] = False
    elif case == 'missing_control':
        t['results'].pop('control48')
    elif case == 'scope':
        t['pi_tested'] = True
    else:
        t['eligible_for_full_scene_comparison'] = False
    with pytest.raises(ValueError):
        eligible_names(t)


def test_missing_admission_pair_cannot_start_development():
    with pytest.raises(ValueError):
        development_names(['admit47'], {'admit47': dict(nonregression=True)})
