import pytest

from scripts.audit_joint_balanced_balloon import PANELS, paired_gate


def passing():
    return {p: {f'balanced_admit{s}_current': dict(no_regression=True, effect=dict(gained_targets=[1]))
                for s in (47, 48)} for p in PANELS}


def test_paired_recovery_passes():
    assert paired_gate(passing(), 'current')


@pytest.mark.parametrize('seed', [47, 48])
def test_no_regression_required_in_every_seed_and_panel(seed):
    for panel in PANELS:
        values = passing()
        values[panel][f'balanced_admit{seed}_current']['no_regression'] = False
        assert not paired_gate(values, 'current')


def test_single_seed_recovery_insufficient():
    values = passing()
    for panel in values.values():
        panel['balanced_admit48_current']['effect']['gained_targets'] = []
    assert not paired_gate(values, 'current')


def test_missing_panel_and_unknown_policy_rejected():
    values = passing()
    del values['window2']
    with pytest.raises(ValueError):
        paired_gate(values, 'current')
    with pytest.raises(ValueError):
        paired_gate(passing(), 'unbounded')
