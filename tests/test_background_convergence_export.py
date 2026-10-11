import pytest

from scripts.export_background_convergence import eligible_names


def trial(passed):
    return dict(results={f'{a}{s}': dict(training_gate_pass=f'{a}{s}' in passed)
                         for a in ('positive_only', 'with_background') for s in (47, 48)},
        eligible_arms=['with_background'], eligible_for_full_scene_comparison=True,
        development_evaluated=False, test_evaluated=False, deployment_approved=False, qualification_achieved=False)


def test_export_requires_both_seeds_and_only_selects_passing_arm():
    assert eligible_names(trial({'with_background47', 'with_background48'})) == ['with_background47', 'with_background48']
    with pytest.raises(ValueError):
        eligible_names(trial({'with_background47'}))
    with pytest.raises(ValueError):
        eligible_names(trial(set()))


def test_cannot_override_failed_gate_or_claim_qualification():
    t = trial({'with_background47', 'with_background48'})
    t['eligible_arms'] = ['positive_only']
    with pytest.raises(ValueError):
        eligible_names(t)
    t = trial({'with_background47', 'with_background48'})
    t['qualification_achieved'] = True
    with pytest.raises(ValueError):
        eligible_names(t)


def test_all_arms_pass_and_missing_checkpoint_rejected():
    t = trial({'with_background47', 'with_background48', 'positive_only47', 'positive_only48'})
    t['eligible_arms'] = ['positive_only', 'with_background']
    assert len(eligible_names(t)) == 4
    t['results'].pop('positive_only47')
    with pytest.raises(ValueError):
        eligible_names(t)
