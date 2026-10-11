import pytest

from scripts.audit_reviewed_negative_correction import eligible_policies


def unchanged():
    return {panel: {f'corrected{s}_{policy}': dict(no_regression=True,
        effect=dict(gained_targets=[], count_delta={c: dict(tp=0, fp=0, fn=0)
                                                   for c in ('red_balloon', 'blue_balloon')}))
        for s in (47, 48) for policy in ('current', 'pooled', 'base_first')}
        for panel in ('original', 'window1', 'window2')}


def test_unchanged_is_not_an_improvement():
    assert not any(eligible_policies(unchanged()).values())


def test_paired_false_positive_reduction_can_qualify():
    values = unchanged()
    for seed in (47, 48):
        values['original'][f'corrected{seed}_current']['effect']['count_delta']['red_balloon']['fp'] = -1
    assert eligible_policies(values) == dict(current=True, pooled=False, base_first=False)


def test_one_seed_or_regression_does_not_qualify():
    values = unchanged()
    values['original']['corrected47_current']['effect']['gained_targets'] = ['target']
    assert not eligible_policies(values)['current']
    values['original']['corrected48_current']['effect']['gained_targets'] = ['target']
    values['window2']['corrected47_current']['no_regression'] = False
    assert not eligible_policies(values)['current']


def test_missing_panel_rejected():
    values = unchanged()
    del values['window1']
    with pytest.raises(ValueError):
        eligible_policies(values)
